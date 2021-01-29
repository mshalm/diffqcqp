#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Solver
{
    public:
    Solver();
    double power_iteration(const MatrixXd &A, const double epsilon,const int maxStep);
    VectorXd iterative_refinement(const Ref<const MatrixXd> &A,const VectorXd &b, const double mu_ir,const double epsilon,const int max_iter);
    VectorXd iterative_refinement2(const Ref<const MatrixXd> &A,const VectorXd &b, const double mu_ir,const double epsilon,const int max_iter);
    VectorXd solveQP( MatrixXd P, const VectorXd &q, const VectorXd &warm_start , const double epsilon, const double mu_prox, const int max_iter, const bool adaptative_rho);
    VectorXd dualFromPrimalQP(const MatrixXd &P,const VectorXd &q,const VectorXd &l, const double &epsilon);
    VectorXd solveDerivativesQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon);
    //VectorXd prox_circle(VectorXd l, const VectorXd &l_n);
    void prox_circle(VectorXd &l, const VectorXd &l_n);
    VectorXd solveQCQP( MatrixXd P, const VectorXd &q, const VectorXd &l_n, const VectorXd &warm_start, const double epsilon, const double mu_prox, const int max_iter,const bool adaptative_rho);
    VectorXd dualFromPrimalQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const double &epsilon);
    VectorXd solveDerivativesQCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l_n, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon);
    std::tuple<MatrixXd,MatrixXd> getE12QCQP(const VectorXd &l_n, const VectorXd &mu, const VectorXd &gamma);

    void prox_lorentz(VectorXd &l);
    VectorXd solveLCQP( MatrixXd P, const VectorXd &q, const VectorXd &warm_start, const double epsilon, const double mu_prox, const int max_iter,const bool adaptative_rho);
    VectorXd dualFromPrimalLCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const double &epsilon);
    VectorXd solveDerivativesLCQP(const MatrixXd &P, const VectorXd &q, const VectorXd &l, const VectorXd &gamma, const VectorXd &grad_l, const double &epsilon);
    
    int test();

    private:
    int prob_id;
};