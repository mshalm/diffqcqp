#include <iostream>
#include "Solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <chrono>

namespace py = pybind11;

using namespace Eigen;
using namespace std;


VectorXd solveQP( const py::EigenDRef<const MatrixXd> &P, const py::EigenDRef<const VectorXd> &q, const py::EigenDRef<const VectorXd> &warm_start,const double epsilon =1e-10, const double mu_prox = 1e-7, const int max_iter=1000, const bool adaptative_rho=true){
    Solver solver;
    VectorXd solution(q.size());
    solution = solver.solveQP(P,q,warm_start,epsilon,mu_prox,max_iter,adaptative_rho);
    return solution;
}

VectorXd solveDerivativesQP(const py::EigenDRef<const MatrixXd> &P, const py::EigenDRef<const VectorXd> &q, const py::EigenDRef<const VectorXd> &l, const py::EigenDRef<const VectorXd> &grad_l, const double epsilon =1e-10){
    Solver solver;
    VectorXd gamma(l.size()),bl(l.size());
    gamma = solver.dualFromPrimalQP(P,q,l,epsilon);
    bl = solver.solveDerivativesQP(P,q,l,gamma,grad_l,epsilon);
    return bl;
}

VectorXd solveQCQP( const py::EigenDRef<const MatrixXd> &P, const py::EigenDRef<const VectorXd> &q,const py::EigenDRef<const VectorXd> &l_n, const py::EigenDRef<const VectorXd> &mu, const py::EigenDRef<const VectorXd> &warm_start,const double epsilon=1e-10,const double mu_prox = 1e-7, const int max_iter = 1000, const bool adaptative_rho = true){
    Solver solver;
    VectorXd solution(q.size()), mul_n(l_n.size());
    mul_n = l_n.cwiseProduct(mu);
    solution = solver.solveQCQP(P,q,mul_n,warm_start,epsilon,mu_prox,max_iter,adaptative_rho);
    return solution;
}

std::tuple<MatrixXd,MatrixXd,VectorXd> solveDerivativesQCQP(const py::EigenDRef<const MatrixXd> &P, const py::EigenDRef<const VectorXd> &q, const py::EigenDRef<const VectorXd> &l_n, const py::EigenDRef<const VectorXd> &mu, const py::EigenDRef<const VectorXd> &l, const py::EigenDRef<const VectorXd> &grad_l, const double epsilon =1e-10){
    Solver solver;
    MatrixXd E1(l_n.size(),l_n.size()),E2(l_n.size(),l_n.size());
    VectorXd mul_n(l_n.size()),gamma(l.size()),blgamma(l.size());
    mul_n = l_n.cwiseProduct(mu);
    gamma = solver.dualFromPrimalQCQP(P,q,mul_n,l,epsilon);
    std::tie(E1,E2) = solver.getE12QCQP(l_n, mu, gamma);
    blgamma = solver.solveDerivativesQCQP(P,q,mul_n,l,gamma,grad_l,epsilon);
    return std::make_tuple(E1,E2,blgamma);
}

VectorXd solveLCQP( const py::EigenDRef<const MatrixXd> &P, const py::EigenDRef<const VectorXd> &q, const py::EigenDRef<const VectorXd> &warm_start,const double epsilon=1e-10,const double mu_prox = 1e-7, const int max_iter = 1000, const bool adaptative_rho = true){
    Solver solver;
    VectorXd solution(q.size());
    solution = solver.solveLCQP(P,q,warm_start,epsilon,mu_prox,max_iter,adaptative_rho);
    return solution;
}

VectorXd solveDerivativesLCQP(const py::EigenDRef<const MatrixXd> &P, const py::EigenDRef<const VectorXd> &q, const py::EigenDRef<const VectorXd> &l, const py::EigenDRef<const VectorXd> &grad_l, const double epsilon =1e-10){
    Solver solver;
    return solver.solveDerivativesLCQP(P,q,l,grad_l,epsilon);
}



PYBIND11_MODULE(diffsolvers, m) {
    m.doc() = "module solving QCQP and QP with ADMM, and computing the derivatives of the solution using implicit differentiation of KKT optimality conditions";
    m.def("solveQP", &solveQP, "A function which solves a QP problem with a regularized ADMM algorithm",py::arg("P"), py::arg("q"),py::arg("warm_start"),py::arg("epsilon") = 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000,py::arg("adaptative_rho")= true, py::return_value_policy::reference_internal );
    m.def("solveQCQP", &solveQCQP, "A function which solves a QCQP problem with a regularized ADMM algorithm",py::arg("P"), py::arg("q"), py::arg("l_n"),py::arg("mu"), py::arg("warm_start"),py::arg("epsilon")= 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000,py::arg("adaptative_rho")= true, py::return_value_policy::reference_internal );
    m.def("solveLCQP", &solveLCQP, "A function which solves a QCQP, Lorentz-cone-constrained problem with regularized ADMM algorithm",py::arg("P"), py::arg("q"), py::arg("warm_start"),py::arg("epsilon")= 1e-10,py::arg("mu_prox")= 1e-7,py::arg("max_iter")= 1000,py::arg("adaptative_rho")= true, py::return_value_policy::reference_internal );
    m.def("solveDerivativesQP", &solveDerivativesQP, "A function which solves the differentiated KKT system of a QP",py::arg("P"), py::arg("q"), py::arg("l"), py::arg("grad_l"), py::arg("epsilon")=1e-10);
    m.def("solveDerivativesQCQP", &solveDerivativesQCQP, "A function which solves the differentiated KKT system of a QCQP",py::arg("P"), py::arg("q"), py::arg("l_n"),py::arg("mu"), py::arg("l"), py::arg("grad_l"), py::arg("epsilon")=1e-10 );
    m.def("solveDerivativesLCQP", &solveDerivativesLCQP, "A function which solves the differentiated KKT system of a Lorentz-cone-constrained QCQP",py::arg("P"), py::arg("q"), py::arg("l"), py::arg("grad_l"), py::arg("epsilon")=1e-10 );

}
