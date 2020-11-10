#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include "Solver.hpp"

using namespace std;
using namespace Eigen;

Solver::Solver(){
    prob_id = 1;
}

VectorXd Solver::iterative_refinement(const MatrixXd &A,const VectorXd &b,const double &mu_ir = 1e-7,const double &epsilon = 1e-10,const int &max_iter = 10){
    VectorXd Ab(A.cols()), delta(A.cols());
    MatrixXd AA_tild(A.cols(), A.cols()),AA_tild_inv(A.cols(), A.cols());
    VectorXd x = VectorXd::Zero(A.cols());
    Ab = A.transpose()*b;
    //MatrixXd A_t = A.transpose();
    AA_tild = A.transpose()*A;
    AA_tild = A.transpose()*A + mu_ir*MatrixXd::Identity(AA_tild.rows(),AA_tild.cols());
    //MatrixXd Id = MatrixXd::Identity(AA_tild.rows(),AA_tild.cols());
    //AA_tild_inv = AA_tild+mu_ir*Id;
    //SelfAdjointEigenSolver<MatrixXd> eigensolver(AA_tild);
    //if (eigensolver.info() != Success) abort();
    //std::cout << " A_t  : " << A_t << std::endl;
    //std::cout << " A  : " << A << std::endl;
    //std::cout << " dd  : " << b << std::endl;
    //cout << "The eigenvalues of AA are:\n" << eigensolver.eigenvalues() << endl;
    //AA_tild_inv = AA_tild_inv.inverse();
    //AA_tild_inv = AA_tild_inv.llt().solve(Id);
    AA_tild_inv.setIdentity();
    AA_tild.llt().solveInPlace(AA_tild_inv);
    int not_improved = 0;
    double res;
    double res_pred = std::numeric_limits<double>::max();
    VectorXd AA_tild_invAb = AA_tild_inv*Ab;
    for(int i = 0; i<max_iter; i++){
        x = mu_ir*AA_tild_inv*x + AA_tild_invAb;
        delta.noalias() = AA_tild*x - Ab;
        //std::cout << "IR res: "<< delta.norm() << std::endl;
        res = delta.norm();
        if(res_pred - res < epsilon){
            not_improved++;
        }
        else{
            res_pred = res;
            not_improved = 0;
        }
        if (res<epsilon || not_improved ==2){
            break;
        }
    }
    //std::cout << " b  : " << x << std::endl;
    return x;
}

double Solver::power_iteration(const MatrixXd &A,const  double &epsilon = 1e-10, const int &max_iter = 100){
    VectorXd v(A.cols()), Av(A.cols());
    v = VectorXd::Random(A.cols());
    v.normalize();
    for (int i =0; i< max_iter; i++){
        Av.noalias() = A*v;
        v = Av;
        v.normalize();
    };
    double l_max;
    Av.noalias() = A*v;
    l_max = v.dot(Av);
    return l_max;
}

VectorXd Solver::solveQP(const MatrixXd &P, const VectorXd &q, const VectorXd &warm_start, const double &epsilon =1e-10, const double &mu_prox = 1e-7, const int &max_iter=1000,const bool &adaptative_rho=true){
    /*typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();*/
    double rho, res_dual, res_prim, mu_thresh, tau_inc, tau_dec;
    mu_thresh = 10.;tau_inc = 2.; tau_dec = 2.;
    MatrixXd Pinv(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size()),Plqu(q.size());
    VectorXd u = VectorXd::Zero(q.size());
    VectorXd l_2 = VectorXd::Zero(q.size());
    l = warm_start;
    //auto t1 = Time::now();
    rho = Solver::power_iteration(P,epsilon, 10);
    rho = std::sqrt(mu_prox*rho)*std::pow(rho/mu_prox,.1);
    std::cout << rho << std::endl;
    q_prox = q;
    MatrixXd Id = MatrixXd::Identity(P.rows(), P.cols());
    Pinv = P+(rho+mu_prox)*Id;
    //auto t2 = Time::now();
    //Pinv = Pinv.inverse();
    //Pinv = PartialPivLU<MatrixXd>(Pinv).inverse();
    Pinv = Pinv.llt().solve(Id); //allocation dynamique ? 
    //std::cout << Pinv*(P+(rho+mu_prox)*Id) << std::endl;
    //auto t3 = Time::now();
    int rho_up = 0;
    std::vector<double> rhos;
    for(int i = 0; i< max_iter; i++){
        l.noalias() = Pinv*(rho*l_2-u-q_prox);
        q_prox = q - mu_prox*l;
        l_2 = (l+u/rho).cwiseMax(0);
        u.noalias() += rho*(l-l_2);
        Plqu.noalias() = P*l+q+u;
        res_dual = Plqu.lpNorm<Infinity>();
        res_prim = (l_2-l).lpNorm<Infinity>();
        std::cout<< "res dual  : "  << res_dual << std::endl;
        if(res_dual < epsilon){
            std::cout<< "num iter QP : "  << i << std::endl;
            std::cout << "rhos :";
            for (int j = 0; j< rhos.size(); j++ ){
                std::cout << rhos.back() <<std::endl;
                rhos.pop_back();
            }
            break;
        }
        if(false && res_prim > mu_thresh*res_dual){
            if (rho_up ==-1){
                tau_inc = 1+.8*(tau_inc-1);
            }
            rho = rho*tau_inc;
            std::cout<< "res dual  : "  << res_dual << " res primal  : "  << res_prim << " rho  : "  << rho << std::endl;
            rhos.push_back(rho);
            Pinv = P+(rho+mu_prox)*Id;
            Pinv = Pinv.llt().solve(Id);
            rho_up= 1;
        }
        else if (false && res_dual > mu_thresh*res_prim){
            if (rho_up ==1){
                tau_dec = 1+.8*(tau_dec-1);
            }
            rho = rho/tau_dec;
            std::cout<< "res dual  : "  << res_dual << " res primal  : "  << res_prim << " rho  : "  << rho << std::endl;
            rhos.push_back(rho);
            Pinv = P+(rho+mu_prox)*Id;
            Pinv = Pinv.llt().solve(Id);
            rho_up=-1;
        }
    };
    /*auto t4 = Time::now();
    fsec fs1 = t4 - t3;
    fsec fs3 = t3 - t2;
    fsec fs4 = t2 - t1;
    fsec fs2 = t4 - t0;
    std::cout << "power iter: " << fs4.count() << "s\n";
    std::cout << "inverting: " << fs3.count() << "s\n";
    std::cout << "iterations: " << fs1.count() << "s\n";
    std::cout << "total: " << fs2.count() << "s\n";*/
    //std::cout << l_2 << std::endl;
    return l_2;
}

VectorXd Solver::dualFromPrimalQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const double &epsilon=1e-10){
    VectorXd gamma(l.size());
    gamma = -(P*l + q);
    for (int i = 0; i<gamma.size();i++){
        if(l(i)>epsilon){
            gamma(i) = 0;
        }
    }
    return gamma;
}

VectorXd Solver::solveDerivativesQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon){
    std::vector<int> not_null;
    for (int i = 0; i<gamma.size(); i++){
        if(gamma(i)>1e-10){
            not_null.push_back(i);
        }
    }
    MatrixXd B = gamma.asDiagonal();
    MatrixXd C = MatrixXd::Identity(l.size(), l.size());
    MatrixXd A_tild(not_null.size(), not_null.size()), B_tild(not_null.size(),l.size()), C_tild(l.size(),not_null.size()), D_tild(l.size(),l.size());
    //need to initialize with zeros
    A_tild = MatrixXd::Zero(not_null.size(), not_null.size());
    for(int i = 0; i< not_null.size();i++){
        A_tild(i,i) = l(not_null[i]);
        B_tild.row(i) = B.row(not_null[i]);
        C_tild.col(i) = C.col(not_null[i]);
    }
    D_tild = P;
    MatrixXd A(l.size()+not_null.size(),l.size()+not_null.size());
    A.topLeftCorner(not_null.size(),not_null.size()) = A_tild;
    A.topRightCorner(not_null.size(),l.size()) = B_tild;
    A.bottomLeftCorner(l.size(),not_null.size()) = C_tild;
    A.bottomRightCorner(l.size(),l.size()) = D_tild;
    A.transposeInPlace();
    VectorXd dd(A.cols());
    for(int i = 0 ; i< dd.size(); i++){
        if(i<not_null.size()){
            dd(i) = 0.;
        }
        else{
            dd(i) = grad_l(i-not_null.size());
        }
    }
    VectorXd b(A.cols());
    b = Solver::iterative_refinement(A,dd);
    VectorXd bl(l.size());
    for(int i = 0; i <l.size();i++){
        bl(i) = b(not_null.size()+i);
    }
    return bl;
}

void Solver::prox_circle(VectorXd &l, const VectorXd &l_n){
    int nb_contacts;
    double norm_l2d;
    VectorXd l_2d(2);
    nb_contacts = l_n.size();
    for(int i = 0; i<nb_contacts; i++){
        l_2d(0) = l(2*i);
        l_2d(1) = l(2*i+1);
        norm_l2d = l_2d.norm();
        if(norm_l2d> l_n(i)){
            l(2*i) = l_2d(0)*l_n(i)/norm_l2d;
            l(2*i+1) = l_2d(1)*l_n(i)/norm_l2d;
        }
    }
    //return l;
}

VectorXd Solver::solveQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &warm_start, const double &epsilon=1e-10, const double &mu_prox = 1e-7, const int &max_iter = 1000){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();

    double rho, res_dual, res_prim, eps_rel, tau_dec, tau_inc, mu_thresh, alpha_relax;
    tau_dec = 2.; tau_inc = 2.; mu_thresh = 10.; alpha_relax = 1.5;
    eps_rel = 1e-4;
    MatrixXd Pinv(P.rows(), P.cols());
    VectorXd q_prox(q.size()),l(q.size()), Plqu (q.size());
    VectorXd u = VectorXd::Zero(q.size());
    VectorXd l_2 = VectorXd::Zero(q.size());VectorXd l_2_pred = VectorXd::Zero(q.size());
    l = warm_start;
    rho = Solver::power_iteration(P,epsilon, 100);
    //double l_min = mu_prox + rho + Solver::power_iteration(P-rho*MatrixXd::Identity(P.rows(),P.cols()),epsilon,10000);
    //std::cout << "lmin : " << l_min << std::endl;
    rho = std::sqrt(mu_prox*rho)*std::pow(rho/mu_prox,.4);
    //rho = std::sqrt(l_min*rho)*std::pow(rho/l_min,.4)*1e-3;
    q_prox = q;
    MatrixXd Id = MatrixXd::Identity(P.rows(), P.cols());
    Pinv = P+(rho+mu_prox)*Id;
    //Pinv = Pinv.inverse();
    //Pinv = PartialPivLU<MatrixXd>(Pinv).inverse();
    Pinv = Pinv.llt().solve(Id); // allocation dynamique ? 
    auto t1 = Time::now();
    std::vector<double> rhos;
    int rho_up = 0;
    //VectorXd solution  = -P.inverse()*q;
    for(int i = 0; i< max_iter; i++){
        l = Pinv*(rho*l_2-u-q_prox);
        q_prox = q - mu_prox*l;
        //l_2 = l+u/rho;
        l_2 = alpha_relax*l + (1-alpha_relax)*l_2+u/rho;
        Solver::prox_circle(l_2,l_n);
        //l_2 = Solver::prox_circle(l+u/rho,l_n);
        //u += rho*(l-l_2);
        u += rho*(alpha_relax*l + (1-alpha_relax)*l_2_pred-l_2);
        Plqu.noalias() = P*l+q+u;
        res_dual = Plqu.norm();
        //Plqu = l_2-l_2_pred;
        //res_dual = rho*Plqu.norm();
        //res = Plqu.lpNorm<Infinity>();
        res_prim = (l_2-(alpha_relax*l + (1-alpha_relax)*l_2_pred)).norm();
        l_2_pred = l_2;
        //std::cout << "res dual : " << res_dual << " res prim : "<< res_prim << "\n";
        //std::cout << "rho : " << rho << std::endl;
        //if(res < epsilon){
        if( res_prim < std::sqrt(l_n.size())*epsilon + eps_rel*l.norm() && res_dual < std::sqrt(q.size())*epsilon ){//+ eps_rel * u.norm()){
            //std::cout << "num iter: " << i << "\n";
            break;
        }
        if( res_prim > mu_thresh*res_dual){
            if (rho_up ==-1){
                tau_inc = 1+.8*(tau_inc-1);
            }
            rho = rho*tau_inc;
            rhos.push_back(rho);
            Pinv = P+(rho+mu_prox)*Id;
            Pinv = Pinv.llt().solve(Id);
            rho_up= 1;
        }
        else if ( res_dual > mu_thresh*res_prim){
            if (rho_up ==1){
                tau_dec = 1+.8*(tau_dec-1);
            }
            rho = rho/tau_dec;
            rhos.push_back(rho);
            Pinv = P+(rho+mu_prox)*Id;
            Pinv = Pinv.llt().solve(Id);
            rho_up=-1;
        }
        /*tau_inc = std::sqrt((res_prim+epsilon)/res_dual); tau_dec = 1/tau_inc;
        if( tau_inc > 5.){
            rho = rho*tau_inc;
            std::cout << "rho: " << rho << "\n";
            Pinv = P+(rho+mu_prox)*Id;
            Pinv = Pinv.llt().solve(Id);
        }
        else if (  tau_dec > 5.){
            rho = rho/tau_dec;
            std::cout << "rho: " << rho << "\n";
            Pinv = P+(rho+mu_prox)*Id;
            Pinv = Pinv.llt().solve(Id);
        }*/
        if (i ==max_iter-1){
            std::cout << "rhos :";
            for (int j = 0; j< rhos.size(); j++ ){
                std::cout << rhos.back() <<std::endl;
                rhos.pop_back();
            }
            std::cout << "res dual : " << res_dual << " res prim : "<< res_prim << "\n";
        }
    };
    auto t2 = Time::now();
    fsec fs1 = t2 - t1;
    fsec fs2 = t2 - t0;
    //std::cout << "iterations: " << fs1.count() << "s\n";
    //std::cout << "total: " << fs2.count() << "s\n";

    return l;
}

VectorXd Solver::dualFromPrimalQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const double &epsilon=1e-10){
    VectorXd gamma(l_n.size()),slack(l_n.size()),l_2d(2) ;
    MatrixXd  A = MatrixXd::Zero(l.size(), l_n.size());
    std::vector<int> not_null;
    slack = l_n;
    for(int i = 0; i<l_n.size(); i++){
        //A(2*i,i) = 2*l(2*i);
        A(2*i,i) = 2*l(2*i);
        //A(2*i+1,i) = 2*l(2*i+1);
        A(2*i+1,i) = 2*l(2*i+1);
    }
    for (int i = 0; i<gamma.size();i++){
        l_2d(0) = l(2*i);
        l_2d(1) = l(2*i+1);
        slack(i) += -l_2d.norm();
        if(slack(i)>epsilon){
            gamma(i) = 0;
        }
        else
        {
            not_null.push_back(i);
        }
    }
    MatrixXd A_tild(A.rows(), not_null.size());
    for (int i = 0; i < not_null.size(); ++i) {
        A_tild.col(i) = A.col(not_null[i]);
    }
    //A_tild = A(all,not_null);
    //A_tild = (A_tild.transpose()*A_tild).inverse()*A_tild.transpose();
    VectorXd gamma_not_null(not_null.size());
    //A_tild = (A_tild.transpose()*A_tild).llt().solve(MatrixXd::Identity(A_tild.cols(),A_tild.cols()))*A_tild.transpose();
    //gamma_not_null = -A_tild*(P*l+q);
    gamma_not_null = -(A_tild.transpose()*A_tild).llt().solve(A_tild.transpose()*(P*l+q));
    int idx;
    for(int i=0; i<not_null.size();i++){
        idx =not_null[i];
        gamma(idx) = gamma_not_null(i);
    }
    return gamma;
}

VectorXd Solver::solveDerivativesQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon){
    int nb_contacts = l_n.size();
    VectorXd slack(nb_contacts);
    slack = -l_n.cwiseProduct(l_n);
    double norm_l2d;
    VectorXd l_2d(2);
    MatrixXd C(2*nb_contacts,nb_contacts);
    MatrixXd D_tild = MatrixXd::Zero(2*nb_contacts, 2*nb_contacts);
    for(int i = 0; i<nb_contacts; i++){
        l_2d(0) = l(2*i);
        l_2d(1) = l(2*i+1);
        norm_l2d = l_2d.squaredNorm();
        slack(i) = slack(i) + norm_l2d;
        C(2*i,i) = 2*l(2*i);
        C(2*i+1,i) = 2*l(2*i+1);
        D_tild(2*i,2*i) = 2*gamma(i);
        D_tild(2*i+1,2*i+1) = 2*gamma(i);
    }
    //std::cout << " l : " << l<< " gamma : " << gamma << std::endl;
    std::vector<int> not_null;
    for (int i = 0; i<nb_contacts; i++){
        if(slack(i)>-1e-10){
            not_null.push_back(i);
        }
    }
    MatrixXd A_tild = MatrixXd::Zero(not_null.size(),not_null.size());
    MatrixXd B = gamma.asDiagonal()*(C.transpose());
    MatrixXd B_tild(not_null.size(),B.cols()), C_tild(C.rows(), not_null.size());
    for(int i = 0; i< not_null.size(); i++){
        A_tild(i,i) = slack(not_null[i]);
        B_tild.row(i) = B.row(not_null[i]);
        C_tild.col(i) = C.col(not_null[i]);
    }
    D_tild = D_tild+P;
    MatrixXd A(l.size()+not_null.size(),l.size()+not_null.size());
    A.topLeftCorner(not_null.size(),not_null.size()) = A_tild;
    A.topRightCorner(not_null.size(),l.size()) = B_tild;
    A.bottomLeftCorner(l.size(),not_null.size()) = C_tild;
    A.bottomRightCorner(l.size(),l.size()) = D_tild;
    //std::cout << " not null size  : " << not_null.size() << "l size: "<< l.size()<< "A size: "<< A.cols()<< std::endl;
    //std::cout << " A  : " << A << std::endl;
    A.transposeInPlace();
    
    VectorXd dd(A.cols());
    for(int i = 0 ; i< dd.size(); i++){
        if(i<not_null.size()){
            dd(i) = 0.;
        }
        else{
            dd(i) = grad_l(i-not_null.size());
        }
    }
    VectorXd b(A.cols());
    b = Solver::iterative_refinement(A,dd);
    VectorXd blgamma = VectorXd::Zero(gamma.size()+l.size());
    for(int i = 0; i<b.size();i++){
        if(i<not_null.size()){
            blgamma(not_null[i]) = b(i);
        }
        else{
            blgamma(nb_contacts-not_null.size()+i) = b(i);
        }
    }
    return blgamma;
}

std::tuple<MatrixXd,MatrixXd> Solver::getE12QCQP(const VectorXd &l_n, const VectorXd &mu, const VectorXd &gamma){
    MatrixXd E1 = MatrixXd::Zero(l_n.size(),l_n.size());
    MatrixXd E2 = MatrixXd::Zero(l_n.size(),l_n.size());
    for (int i = 0;i<l_n.size();i++){
        E1(i,i) = 2*gamma(i)*l_n(i)*l_n(i)*mu(i);
        E2(i,i) = 2*gamma(i)*l_n(i)*mu(i)*mu(i);
    }
    return std::make_tuple(E1,E2);
}

int Solver::test(){
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;
    MatrixXd m2(4,4);
    m2 << 4.45434,  1.11359, -2.22717,  1.11359,
         1.11359,  4.45434,  1.11359, -2.22717,
        -2.22717,  1.11359,  4.45434,  1.11359,
         1.11359, -2.22717,  1.11359,  4.45434;
    VectorXd sol(4),sol2(4),q2(4),l_n(2), warm_start2(4);
    l_n(0) = 1;
    l_n(1) = 1;
    l_n = l_n*10000;
    q2 << -0.0112815,-0.0083385,-0.0083385,-0.0112815;
    //q2 << -0.00981,-0.00981,-0.00981,-0.00981;
    warm_start2 << 0.00220234,0.00220234,0.00220234,0.00220234;
    m2.setZero();m2(0,0) = .005;m2(1,1) = 3.;
    q2.setZero(); q2(0) = -8; q2(1) = -5;
    warm_start2.setZero();
    //sol = Solver::solveQCQP(m2,q2,l_n,warm_start2, 1e-10,1e-7,1000);
    sol = Solver::solveQP(m2,q2,warm_start2, 1e-10,1e-7,1000);
    std::cout << " solution  : " << sol << std::endl;
    VectorXd gamma(sol.size());
    gamma = Solver::dualFromPrimalQCQP(m2,q2,l_n,sol);
    MatrixXd  A(sol.size(), l_n.size());
    for(int i = 0; i<l_n.size(); i++){
        A(2*i,i) = 2*sol(2*i);
        A(2*i+1,i) = 2*sol(2*i+1);
    }
    //std::cout<< "KKT : " << m2*sol + q2 + A*gamma << std::endl;
    //std::cout << "solution : " << sol << std::endl;
    //std::cout << "solution dual : " << gamma << std::endl;
    VectorXd grad_l(sol.size());
    VectorXd bl(sol.size());
    bl = Solver::solveDerivativesQCQP(m2,q2,l_n,sol,gamma,grad_l,1e-8);
    //std::cout << " bl : " << bl << std::endl;

    sol2 = Solver::solveQP(m2,q2,warm_start2,1e-10,1e-7,1);
    VectorXd gamma2(sol2.size());
    gamma2 = Solver::dualFromPrimalQP(m2,q2,sol2);
    //std::cout<< "KKT 2: " << m2*sol2 + q2 + gamma2 << std::endl;
    //std::cout << "solution 2: " << sol2 << std::endl;
    //std::cout << "solution dual 2 : " << gamma2 << std::endl;
    VectorXd grad_l2(sol2.size());
    VectorXd bl2(sol2.size());
    bl2 = Solver::solveDerivativesQP(m2,q2,sol2,gamma2,grad_l2,1e-8);
    //std::cout << " bl2 : " << bl2 << std::endl;

    
    MatrixXd G2(12,12);
    G2 <<  6.6174e-24,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.8452e-04, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
         0.0000e+00, -6.6174e-24,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00, -3.9642e-04,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00, -6.6174e-24,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.9642e-04, -7.1925e-20, 0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  6.6174e-24,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.8452e-04,  4.4048e-20,
        -1.0544e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  4.3570e+03, -1.6704e+00,  4.4543e+00,  1.6704e+00,  1.1136e+00,  1.6704e+00, 1.1136e+00, -1.6704e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.6704e+00, 4.3570e+03, -1.6704e+00,  1.1136e+00,  1.6704e+00,  1.1136e+00, 1.6704e+00,  4.4543e+00,
         0.0000e+00, -1.0544e+00,  0.0000e+00,  0.0000e+00,  4.4543e+00, -1.6704e+00,  5.3243e+03,  1.6704e+00,  1.1136e+00,  1.6704e+00, 1.1136e+00, -1.6704e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  1.6704e+00, 1.1136e+00,  1.6704e+00,  5.3243e+03, -1.6704e+00,  4.4543e+00, -1.6704e+00,  1.1136e+00,
        0.0000e+00,  0.0000e+00, -1.0544e+00,  0.0000e+00,  1.1136e+00, 1.6704e+00,  1.1136e+00, -1.6704e+00,  5.3243e+03, -1.6704e+00,4.4543e+00,  1.6704e+00,
        0.0000e+00,  0.0000e+00, -1.9131e-16,  0.0000e+00,  1.6704e+00,1.1136e+00,  1.6704e+00,  4.4543e+00, -1.6704e+00,  5.3243e+03,-1.6704e+00,  1.1136e+00,
        0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0544e+00,  1.1136e+00,1.6704e+00,  1.1136e+00, -1.6704e+00,  4.4543e+00, -1.6704e+00,4.3570e+03,  1.6704e+00,
        0.0000e+00,  0.0000e+00,  0.0000e+00,  9.5861e-17, -1.6704e+00,4.4543e+00, -1.6704e+00,  1.1136e+00,  1.6704e+00,  1.1136e+00,1.6704e+00,  4.3570e+03;
    G2(0,0) = 0; G2(1,1) = 4e1;
    G2 = G2*G2.transpose();
    VectorXd g2(12), l_ng(6), l_ng2(6);
    g2 << 0.0000e+00,0.0000e+00,0.0000e+00,0.0000e+00,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14,7.2829e-04,2.2609e-14;
    VectorXd sol3(12);
    //VectorXd warm_start3  = VectorXd::Zero(12);
    VectorXd warm_start3  = VectorXd::Random(12);

    double mean,mean2, mean3;
    mean = 0;mean2 = 0.; mean3 = 0.;
    int ntest = 100;
    std::srand((unsigned int) time(0));
    int test_dimension = 6;
    MatrixXd G(2*test_dimension,2*test_dimension);
    VectorXd g(test_dimension);
    for (int i = 0; i< ntest; i++){
        //std::cout<< "prob id : " << i << std::endl;
        G = MatrixXd::Random(2*test_dimension,2*test_dimension);
        G = G*G.transpose();
        SelfAdjointEigenSolver<MatrixXd> eigensolver(G);
        if (eigensolver.info() != Success) abort();
        //cout << "The eigenvalues of G are:\n" << eigensolver.eigenvalues() << endl;
        g = VectorXd::Random(2*test_dimension);
        l_ng = VectorXd::Random(test_dimension)+VectorXd::Ones(test_dimension);
        l_ng = l_ng*.1;
        l_ng2 = l_ng*100000;
        //std::cout<< "lng: " << l_ng << "\n";
        auto t0 = Time::now();
        //sol3 = Solver::solveQP(G,g,warm_start3,1e-10,1e-7,10000);
        auto t1 = Time::now();
        auto t2 = Time::now();
        sol3 = Solver::solveQCQP(G,g,l_ng,warm_start3,1e-10,1e-7,10000);
        auto t3 = Time::now();
        sol3 = Solver::solveQCQP(G,g,l_ng2,warm_start3,1e-10,1e-7,10000);
        auto t4 = Time::now();
        fsec fs = t1 - t0;
        fsec fs2 = t3 - t2;
        fsec fs3 = t4 - t3;
        mean += fs.count();
        mean2 += fs2.count();
        mean3 += fs3.count();
        //std::cout << "grad to sol: " << (G*sol3 +g ) << "\n";

    }
    
    //fsec fs = t1 - t0;
    //std::cout<< "solving QP: " << mean/ntest << "s\n";
    std::cout<< "solving QCQP1: " << mean2/ntest << "s\n";
    std::cout<< "solving QCQP2: " << mean3/ntest << "s\n";

    //std::cout << " IR : " << Solver::iterative_refinement(G,g,1e-14) << std::endl;
    return 0;
}