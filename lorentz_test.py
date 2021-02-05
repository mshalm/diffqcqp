import numpy as np
import scipy
from time import time
import timeit
from tqdm import tqdm
from torch import optim
from qcqp import LCQPFn2
import torch
torch.set_default_dtype(torch.double)
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from qpth.qp import QPFunction
import matplotlib.pyplot as plt
import osqp
plt.style.use('bmh')
import pdb

class QCQP_cvxpy(nn.Module):
    def __init__(self,NC, reg, eps=1e-14, max_iter = 100):
        '''
        '''
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.NC = NC
        self.reg = reg
        self.initLayer()

    def initLayer(self):
        NC = self.NC
        N = NC * 3
        l = cp.Variable(N)
        A = cp.Parameter((N, N),nonneg= True)
        b = cp.Parameter(N)
        
        #Gs = []
        #for i in range(N//2):
        constraints = [cp.SOC(l[i], l[(NC+2*i):(NC+2*(i+1))]) for i in range(NC)]
        #constraints = []
        #objective = cp.Minimize(0.5 * cp.quad_form(l_t,A) + b.T@l_t )
        objective = cp.Minimize(0.5 * cp.sum_squares(A@l) + b.T@l )
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        self.cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[l])
    
    def forward(self,P_sqrt,q):
        
        # solve the problem
        k=1e-11 #regularization of P to get Cholesky's decomposition 
        k=0.
        b = -P_sqrt.transpose(1,2).bmm(q)
        #L = torch.transpose(torch.cholesky(P+k*torch.eye(P.size()[1])),1,2)
        #print(L.size(), q.size(), mu.size(), l_n.size())
        solution, = self.cvxpylayer(P_sqrt + self.reg * torch.eye(q.size()[1]),b.squeeze(2),solver_args={'eps': self.eps,'max_iters':self.max_iter})
        #solution, = cvxpylayer(P,q.squeeze(2),(mu*l_n).squeeze(2),solver_args={'eps': self.eps,'max_iters':self.max_iter})
        #print(solution)
        return solution




cvxpy_time = {'forward': [], 'backward':[], 'fnan': 0, 'bnan': 0, 'obj': [], 'grad': [], 'drop': []}
qcqp_time = {'forward': [], 'backward':[], 'fnan': 0, 'bnan': 0, 'obj': [], 'grad': [], 'drop': []}
comp = {'abs_err': [], 'rel_err': [], 'abs_err_grad': [], 'rel_err_grad': [], 'our_advantage': [], 'our_rel_advantage': []}

# number of tests
n_testqcqp = 400

# number of Lorentz cones
NC = 8

# number of problem variables
N = NC * 3

# singular values of objective matrix range from 10 ^ (-scale) to 10 ^ scale
scale = 3

# singular values are randomly set to zero with probability zero_chance
zero_chance = 0.0

# cvxpy requires strict convexity; cvxpy_reg * eye(N) is added to the obejctive matrix to ensure this
cvxpy_reg = 0.0 #10 ** (-2.5)
qcqp2 = QCQP_cvxpy(NC, cvxpy_reg, eps=1e-10,max_iter = 1000000)

def qcqpfunct(P_sqrt, q):
    warm_start = torch.rand(q.size())
    #pdb.set_trace()
    P = P_sqrt.transpose(1,2).bmm(P_sqrt)
    b = -P_sqrt.transpose(1,2).bmm(q)
    l1 = LCQPFn2().apply(P, b, warm_start, 1e-10, 1000000)
    return l1

def cvxpyfunct(P_sqrt, q):
    warm_start = torch.rand(q.size())
    return qcqp2(P_sqrt,q).unsqueeze(2)

def lossfun(l, P_sqrt, q):
    return 0.5 * torch.sum((P_sqrt.bmm(l) - q) ** 2)

def QCQP_eval(P_sqrt, q, func, timed):
    P_sqrt_qcqp = torch.nn.parameter.Parameter(P_sqrt.detach().clone(), requires_grad= True)
    q_qcqp = torch.nn.parameter.Parameter(q.detach().clone(), requires_grad= True)
    lr = 1e-4
    #pdb.set_trace()
    optimizer_qcqp = optim.Adam([P_sqrt_qcqp,q_qcqp], lr=lr)
    #optimizer_qcqp.zero_grad()
    l1 = func(P_sqrt_qcqp, q_qcqp)
    fnan = 0
    if np.any(np.isnan(l1.clone().detach().numpy())):
        fnan += 1
    tf = 0
    if timed:
        tf = timeit.timeit(lambda: func(P_sqrt_qcqp,q_qcqp),number = 10)/10.
    L1 = lossfun(l1, P_sqrt_qcqp, q_qcqp)
    if L1 < 0:
        pdb.set_trace()
    loss = L1.clone().detach().numpy().item()
    
    tb = 0
    if timed:
        tb = timeit.timeit(lambda:L1.backward(retain_graph=True),number = 10)/10.
    L1.backward()
    #pdb.set_trace()
    bnan = 0
    if np.any(np.isnan(P_sqrt_qcqp.grad.detach().numpy())) or np.any(np.isnan(q_qcqp.grad.detach().numpy())):
        bnan += 1
    grad = torch.cat((P_sqrt_qcqp.grad,q_qcqp.grad),dim=2).detach().numpy()
    optimizer_qcqp.step()
    l1 = func(P_sqrt_qcqp,q_qcqp)
    L1 = lossfun(l1, P_sqrt_qcqp, q_qcqp)
    drop = loss - L1.detach().numpy().item()
    return (fnan, bnan, tf, tb, loss, grad, drop)

for i in tqdm(range(n_testqcqp)):    
    #P = torch.rand((1,8,8),dtype = torch.double)
    P_sqrt = torch.rand(N) - 0.5
    P_sqrt = P_sqrt * scale
    P_sqrt = torch.pow(10, P_sqrt)
    P_sqrt[torch.rand(N) < zero_chance] = 0.0
    P_sqrt = torch.diag(P_sqrt).unsqueeze(0)
    #pdb.set_trace()
    q = torch.rand((1,N,1),dtype = torch.double) - 0.5
    q_pow = torch.rand((1,N,1),dtype = torch.double) - 0.5
    q_pow = torch.pow(10, q_pow)
    q = q * q_pow
    #P = torch.matmul(P, torch.transpose(P,1,2))
    (fnan, bnan, tf, tb, loss, grad, drop) = QCQP_eval(P_sqrt, q, qcqpfunct, True)
    qcqp_time['fnan'] += fnan
    qcqp_time['bnan'] += bnan
    qcqp_time['forward']+= [tf]
    qcqp_time['backward']+= [tb]
    qcqp_time['grad'] += [grad]
    qcqp_time['obj'] += [loss]
    qcqp_time['drop'] += [drop] 

    (fnan, bnan, tf, tb, loss, grad, drop) = QCQP_eval(P_sqrt, q, cvxpyfunct, True)
    cvxpy_time['fnan'] += fnan
    cvxpy_time['bnan'] += bnan
    cvxpy_time['forward']+= [tf]
    cvxpy_time['backward']+= [tb]
    cvxpy_time['grad'] += [grad]
    cvxpy_time['obj'] += [loss]
    cvxpy_time['drop'] += [drop] 


    comp['abs_err'] += [np.abs(qcqp_time['obj'][-1] - cvxpy_time['obj'][-1])]
    comp['our_advantage'] += [qcqp_time['drop'][-1] - cvxpy_time['drop'][-1]]
    comp['rel_err'] += [2 * comp['abs_err'][-1]/np.abs(qcqp_time['obj'][-1] + cvxpy_time['obj'][-1])]
    comp['our_rel_advantage'] += [2 * comp['our_advantage'][-1] / (np.abs(qcqp_time['drop'][-1]) + np.abs(cvxpy_time['drop'][-1]))]
    comp['abs_err_grad'] += [np.sqrt(np.sum((cvxpy_time['grad'][-1]-qcqp_time['grad'][-1]) ** 2))]
    cvxpy_time['grad'][-1] = np.sqrt(np.sum(cvxpy_time['grad'][-1] ** 2))
    qcqp_time['grad'][-1] = np.sqrt(np.sum(qcqp_time['grad'][-1] ** 2))
    comp['rel_err_grad'] += [2 * comp['abs_err_grad'][-1]/np.abs(qcqp_time['grad'][-1] + cvxpy_time['grad'][-1])]
    #pdb.set_trace()

    

    #cvxpy_time['forward']+= [t5-t4]
    #cvxpy_time['backward']+= [t7-t6]



cvxpy_time['mean forward'] = sum(cvxpy_time['forward'])/n_testqcqp
cvxpy_time['mean backward'] = sum(cvxpy_time['backward'])/n_testqcqp
qcqp_time['mean forward'] = sum(qcqp_time['forward'])/n_testqcqp
qcqp_time['mean backward'] = sum(qcqp_time['backward'])/n_testqcqp

cvxpy_time['error forward'] = np.std(cvxpy_time['forward'])
cvxpy_time['error backward'] = np.std(cvxpy_time['backward'])
print(cvxpy_time['error forward'],np.max(cvxpy_time['forward']), np.min(cvxpy_time['forward']))
qcqp_time['error forward'] = np.std(qcqp_time['forward'])
print(qcqp_time['error forward'], np.max(qcqp_time['forward']), np.min(qcqp_time['forward']))
qcqp_time['error backward'] = np.std(qcqp_time['backward'])

print('cvx:',cvxpy_time['fnan'],cvxpy_time['bnan'])
print('lcqp:',qcqp_time['fnan'],qcqp_time['bnan'])

barWidth = 0.35


y1 = [cvxpy_time['mean forward'],qcqp_time['mean forward']]
y2 = [cvxpy_time['mean backward'], qcqp_time['mean backward'] ]
er1 = [cvxpy_time['error forward'],qcqp_time['error forward']]
er2 = [cvxpy_time['error backward'],qcqp_time['error backward']]
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]
plt.figure(1)
plt.bar(r1, y1, width = barWidth, color = ['cornflowerblue' for i in y1], linewidth = 2,log = True, label="forward", yerr = er1)
plt.bar(r2, y2, width = barWidth, color = ['coral' for i in y1], linewidth = 4,log = True,label="backward", yerr = er2)
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ['cvxpylayers', 'Ours'])
plt.ylabel('Runtime (s)')
plt.title('QCQP solvers')
plt.ylim(bottom = 1e-5, top= 1e-1)
plt.legend()

plt.figure(2)
hist_vals = np.log10(np.array(comp['abs_err']) + 1e-16)


plt.hist(hist_vals)
plt.xlabel('log10(absolute obj. err)')

plt.figure(3)
hist_vals = np.log10(np.array(comp['rel_err']) + 1e-16)


plt.hist(hist_vals)
plt.xlabel('log10(relative obj. err)')

plt.figure(4)
print(np.array(qcqp_time['obj']).min())
hist_vals = np.log10(np.array(qcqp_time['obj'])+ 1e-16)

plt.hist(hist_vals)
plt.xlabel('log10(objective)')




plt.figure(5)
hist_vals = np.log10(np.array(comp['abs_err_grad']))


plt.hist(hist_vals)
plt.xlabel('log10(absolute grad. err)')

plt.figure(6)
hist_vals = np.log10(np.array(comp['rel_err_grad']))


plt.hist(hist_vals)
plt.xlabel('log10(relative grad. err)')

plt.figure(7)
hist_vals = np.log10(np.array(qcqp_time['grad']))
hist_vals2 = np.log10(np.array(cvxpy_time['grad']))

plt.hist(hist_vals, alpha=0.5,label='ours')
plt.hist(hist_vals2, alpha=0.5,label='cvxpy')
plt.xlabel('log10(frobenius_norm(grad))')
plt.legend(loc='upper right')


plt.figure(8)
hist_vals = np.array(comp['our_advantage'])

plt.hist(hist_vals)
plt.xlabel('(abs. performance difference)')

plt.figure(9)
hist_vals = np.array(comp['our_rel_advantage'])

plt.hist(hist_vals)
plt.xlabel('(rel. performance difference)')

plt.figure(10)
hist_vals = np.log10(np.array(qcqp_time['drop']))
hist_vals2 = np.log10(np.array(cvxpy_time['drop']))

plt.hist(hist_vals, alpha=0.5,label='ours')
plt.hist(hist_vals2, alpha=0.5,label='cvxpy')
plt.xlabel('log10(\Delta loss), (more is better)')
plt.legend(loc='upper right')



plt.show()