// Header includes 
#include <cinttypes>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/DavidsonSymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/Util/SimpleRandom.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>

// Namespace directives and declarations
namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal
using namespace Spectra;

// Type aliases + alias templates 
using std::vector;
using uint_32 = uint_fast32_t;
using Index = Eigen::Index;
template <typename T> 
using Vector = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
using VectorXd = Eigen::VectorXd;
using VectorXf = Eigen::VectorXf;
using VectorXu = Eigen::Matrix< uint_32, Eigen::Dynamic, 1 >;

// Clamps x to the interval [lb, ub] such that: 
// x < lb ==> x = lb
// x > ub ==> x = ub 
// lb <= x <= ub ==> x = x 
constexpr float clamp(const float x, const float lb, const float ub) {
  return(x < lb ? lb : (x > ub ? ub : x));
}

// Linear step from 0 -> 1 
constexpr float smooth_step0(const float x, const float lb, const float ub) {
  return clamp((x - lb) / (ub - lb), 0.0, 1.0);
}
constexpr float smooth_step1(float x, const float lb, const float ub) {
  x = clamp((x - lb) / (ub - lb), 0.0, 1.0);
  return x * x * (3 - 2 * x);
}
constexpr float smooth_step2(float x, const float lb, const float ub) {
  x = clamp((x - lb) / (ub - lb), 0.0, 1.0);
  return x * x * x * (x * (x * 6 - 15) + 10);
}


// Represents the matrix L = (D1 * D1.T) where D1 := vertex/edge oriented boundary matrix
template< typename F = float, bool lex_order = true >
struct UpLaplacian1_lowerstar {
  using Scalar = F;  // required by Eigen 
  using VectorXS = Vector< Scalar >;
  const size_t nv;
  const size_t ne; 
  mutable vector< Scalar > r;  // need mutability to re-use workspace
  const Vector< Scalar > fv;
  const VectorXu I;
  const VectorXu J; 
  UpLaplacian1_lowerstar(VectorXS fv_, VectorXu I_, VectorXu J_) : nv(fv_.size()), ne(I_.size()), fv(fv_), I(I_), J(J_) {
    r = vector< Scalar >(ne, 0.0); // workspace vector 
  }
  size_t rows() const { return nv; }
  size_t cols() const { return nv; } 
  // Matvec operation: Lx |-> y for any vector x
  void perform_op(const Scalar* x, Scalar* y) const {
    // Ensure workplace vectors are zero'ed
    std::fill(r.begin(), r.end(), 0);
    std::fill(y, y+nv, 0);
    // std::fill(rz.begin(), rz.end(), 0);
    
    // r = D1.T @ x 
    Scalar ew; 
    size_t i, j;
    for (size_t cc = 0; cc < ne; ++cc){
      i = I[cc];
      j = J[cc];
      ew = fv[i] >= fv[j] ? fv[i] : fv[j];
      r[cc] = ew*x[i] - ew*x[j];
    }

    // y = D1 @ r
    for (size_t cc = 0; cc < ne; ++cc){
      i = I[cc];
      j = J[cc];
      ew = fv[i] >= fv[j] ? fv[i] : fv[j];
      y[i] += ew*r[cc];
      y[j] -= ew*r[cc];
    }
  }
  // Spectra deduces a member function: Matrix operator* (const Eigen::Ref< const Matrix > &mat_in) const
};

template< typename T, typename F = typename T::Scalar > 
auto LanczosInit(T&& op, const Index nev, const Index ncv, const Index max_iter, const double tol, Vector< F > v0) -> py::dict {
  SymEigsSolver< T > solver(op, nev, ncv);
  solver.init_v0(v0);
  solver.compute_ev(SortRule::LargestMagn, max_iter, tol, SortRule::LargestAlge); // SortRule, maxit, tol, sorting
  Vector< F > evalues;
  std::string status = ""; 
  if (solver.info() == CompInfo::Successful){
    evalues = solver.eigenvalues();
    status = "success";
  } else if (solver.info() == CompInfo::NotConverging){
    status = "not converged";
  } else if (solver.info() == CompInfo::NumericalIssue){
    status = "numerical issues encountered";
  } else {
    status = "failed";
  }
  py::dict res;
  res["eigenvalues"] = evalues;
  res["n_restarts"] = solver.num_iterations();
  res["n_operations"] = solver.num_operations();
  res["v0"] = v0;
  res["nev"] = nev; // number of requested eigenvalues
  res["ncv"] = ncv; // number of Lanczos vectors
  res["ritz_values"] = solver.ritz_values();
  res["ritz_converged"] = solver.ritz_converged();
  res["status"] = status; 
  return(res);
}

template< typename T, typename F = typename T::Scalar >
auto IRL(T&& op, const Index nev, const Index ncv, const Index max_iter = 1000, const F tol = 1e-10) -> py::dict { 
  SimpleRandom< F > rng(0);
  auto v0 = rng.random_vec(op.rows()); //Eigen::Matrix< F, Eigen::Dynamic, 1 >
  return LanczosInit(std::forward<T>(op), nev, ncv, max_iter, tol, v0);
}

auto SparseMatIRL(Eigen::SparseMatrix< double > M, const Index nev, const Index ncv, const Index max_iter, const double tol) -> py::dict { 
  SparseSymMatProd< double > op(M);
  return IRL< SparseSymMatProd< double > >(std::move(op), nev, ncv, max_iter, tol);
}

auto UpLaplacian1_matvec(VectorXf x, VectorXf fv, VectorXu I, VectorXu J) -> VectorXf {
  UpLaplacian1_lowerstar< float > op(fv, I, J);
  VectorXf y = VectorXf::Zero(fv.size());
  op.perform_op(x.data(), y.data());
  return(y);
}

auto UpLaplacian1_IRL(VectorXf fv, VectorXu I, VectorXu J, const Index nev, const Index ncv, const Index max_iter, const float tol) -> py::dict {
  UpLaplacian1_lowerstar< float > op(fv, I, J);
  return IRL< UpLaplacian1_lowerstar< float > >(std::move(op), nev, ncv, max_iter, tol);
};

// clang -Wall -fPIC -c src/pbsig/lanczos_spectra.cpp -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -Iextern/eigen -Iextern/spectra/include
PYBIND11_MODULE(_lanczos, m) {
  m.doc() = "Lanczos multiplication module";
  m.def("sparse_lanczos", &SparseMatIRL, "Implicitly restarted Lanczos on sparse matrix");
  m.def("UL1_LS_matvec", &UpLaplacian1_matvec, "Up-Laplacian1 lower-star matrix-vector multiplication");
  m.def("UL1_LS_lanczos", &UpLaplacian1_IRL, "Up-Laplacian1 lower-star Lanczos (implicit restart)");
  // m.def("sparse_lanczos_init", &SparseLanczosInit, "Implicitly restarted Lanczos on sparse matrix");
  // m.def("UL1_LS_jacobi_davidson", &UpLaplacian1_JacobiDavidson, "A function that adds two numbers");
}