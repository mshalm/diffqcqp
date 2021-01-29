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
    def __init__(self,eps=1e-14, max_iter = 100):
        '''
        '''
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.initLayer()

    def initLayer(self):
        N = 8
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
    
    def forward(self,P,q):
        
        # solve the problem
        k=1e-11 #regularization of P to get Cholesky's decomposition 
        k=0.
        P_sqrt = scipy.linalg.sqrtm(P.detach().numpy().copy()[0,:,:])
        P_sqrt = torch.tensor(P_sqrt).unsqueeze(0)
        #L = torch.transpose(torch.cholesky(P+k*torch.eye(P.size()[1])),1,2)
        #print(L.size(), q.size(), mu.size(), l_n.size())
        solution, = self.cvxpylayer(P_sqrt,q.squeeze(2),solver_args={'eps': self.eps,'max_iters':self.max_iter})
        #solution, = cvxpylayer(P,q.squeeze(2),(mu*l_n).squeeze(2),solver_args={'eps': self.eps,'max_iters':self.max_iter})
        #print(solution)
        return solution




cvxpy_time = {'forward': [], 'backward':[], 'fnan': 0, 'bnan': 0}
qcqp_time = {'forward': [], 'backward':[], 'fnan': 0, 'bnan': 0}
n_testqcqp= 10
NC = 8
N = NC * 3
scale = 8
for i in tqdm(range(n_testqcqp)):    
    #P = torch.rand((1,8,8),dtype = torch.double)
    P = torch.rand(N)*2 -1
    P = P * scale
    P = torch.diag(torch.exp(P)).unsqueeze(0)
    #P = torch.matmul(P, torch.transpose(P,1,2))
    P = torch.nn.parameter.Parameter(P, requires_grad= True)
    q = torch.rand((1,N,1),dtype = torch.double)*2-1
    q = q * scale
    q = torch.nn.parameter.Parameter(q, requires_grad= True)
    lr = 0.1
    optimizer2 = optim.Adam([P,q], lr=lr)
    loss = nn.MSELoss()
    relu = torch.nn.ReLU()
    threshold = nn.Threshold(threshold=1e-5, value =1e-5)
    target = torch.ones(q.size())
    qcqp = LCQPFn2().apply
    #warm_start = torch.zeros(q.size())
    warm_start = torch.rand(q.size())
    t0 = time()
    l1= qcqp(P,q,warm_start,1e-10,1000000)
    if np.any(np.isnan(l1.detach().numpy())):
        qcqp_time['fnan'] += 1
    t1= time()
    qcqp_time['forward']+= [timeit.timeit(lambda:qcqp(P,q,warm_start,1e-10,1000000),number = 10)/10.]
    #pdb.set_trace()
    L1 = loss(l1, target)
    optimizer2.zero_grad()
    qcqp_time['backward']+= [timeit.timeit(lambda:L1.backward(retain_graph=True),number = 10)/10.]
    t2 = time()
    L1.backward()
    t3 = time()

    if np.any(np.isnan(P.grad.detach().numpy())) or np.any(np.isnan(q.grad.detach().numpy())):
        qcqp_time['bnan'] += 1

    #qcqp_time['forward']+= [t1-t0]
    #qcqp_time['backward']+= [t3-t2]
    qcqp2 = QCQP_cvxpy(eps=1e-10,max_iter = 1000000)
    t4 = time()
    l1 = qcqp2(P,q)
    cvxpy_time['forward']+= [timeit.timeit(lambda:qcqp2(P,q),number = 10)/10.]
    if np.any(np.isnan(l1.detach().numpy())):
        cvxpy_time['fnan'] += 1
    t5= time()
    L1 = loss(l1.unsqueeze(2), target)
    optimizer2.zero_grad()
    t6 = time()
    cvxpy_time['backward']+= [timeit.timeit(lambda:L1.backward(retain_graph=True),number = 10)/10.]
    t7 = time()

    if np.any(np.isnan(P.grad.detach().numpy())) or np.any(np.isnan(q.grad.detach().numpy())):
        qcqp_time['bnan'] += 1

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
plt.figure()
plt.bar(r1, y1, width = barWidth, color = ['cornflowerblue' for i in y1], linewidth = 2,log = True, label="forward", yerr = er1)
plt.bar(r2, y2, width = barWidth, color = ['coral' for i in y1], linewidth = 4,log = True,label="backward", yerr = er2)
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ['cvxpylayers', 'Ours'])
plt.ylabel('Runtime (s)')
plt.title('QCQP solvers')
plt.ylim(bottom = 1e-5, top= 1e-1)
plt.legend()
plt.show()