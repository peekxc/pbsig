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

// Discontinuous step that returns 0 for all (-infty, a] and 1 for (a, infty)
constexpr float step(const float x, const float a){
  return(x <= a ? 0.0 : 1.0);
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

template < typename F = float, bool reverse = false >
struct SmoothStep {
  F a, b, eps, w; 
  F operator()(F x) const {
    if constexpr(reverse){
      return 1.0 - smooth_step1(x, a-w, b-eps);
    } else {
      return smooth_step1(x, a-w, b-eps);
    }  
  }
};

// Linear operator representing the 0-th Up Laplacian for vertex-and-edge valued boundary matrices
// coming from lower-star filtrations, at scale (i,j), for some fixed simplicial complex K (w/ indices I, J)
template< typename F = float, bool lex_order = true >
struct UpLaplacian0_VELS {
  using Scalar = F;  // required by Eigen 
  using VectorXS = Vector< Scalar >;
  const size_t nv;
  const size_t ne; 
  const SmoothStep< F, true > sv;
  const SmoothStep< F, false > se;
  mutable vector< Scalar > r;  // need mutability to re-use workspace with const declared member functions
  const Vector< Scalar > fv;   // filter values 
  const VectorXu I;            // edge indices 
  const VectorXu J;            // edge indices 
  Vector< Scalar > Df;         // compute during initialization
  //const F a_, const F b_, const F eps, const F w
  // a(a_), b(b_)
  UpLaplacian0_VELS(VectorXS fv_, VectorXu I_, VectorXu J_, SmoothStep< F, true > _sv, SmoothStep< F, false > _se) : nv(fv_.size()), ne(I_.size()), sv(_sv), se(_se), fv(fv_), I(I_), J(J_) {
    r = vector< Scalar >(ne, 0.0); // workspace vector 
    // smooth_step1()
    // ss_a = smoothstep(lb = a-w, ub = a+eps, reverse = True)     #   1 (a-w) -> 0 (a), includes (-infty, a]
    // ss_b = smoothstep(lb = b-w, ub = b+eps, reverse = True)     #   1 (b-w) -> 0 (b), includes (-infty, b]
    // ss_ac = smoothstep(lb = a-w, ub = a+eps, reverse = False) 
    Df = Vector< Scalar >::Zero(nv);
    Scalar ew = 0.0;
    size_t i, j;
    for (size_t cc = 0; cc < ne; ++cc){
      i = I[cc];
      j = J[cc];
      ew = std::max(fv[i], fv[j]);
      Df[i] += ew * ew;
      Df[j] += ew * ew;
    }
  }
  size_t rows() const { return nv; }
  size_t cols() const { return nv; } 
  // Matvec operation: Lx |-> y for any vector x
  void perform_op(const Scalar* x, Scalar* y) const {
    // Ensure workplace vectors are zero'ed
    std::fill(r.begin(), r.end(), 0);
    std::fill(y, y+nv, 0);

    for (size_t cc = 0; cc < nv; ++cc){
      y[cc] += x[cc] * Df[cc] * fv[cc] * fv[cc];
    }
    Scalar ew; 
    size_t i, j;
    for (size_t cc = 0; cc < ne; ++cc){
      i = I[cc];
      j = J[cc];
      ew = std::max(fv[i], fv[j]);
      y[i] -= x[j] * (ew * ew) * fv[i] * fv[j];
      y[j] -= x[i] * (ew * ew) * fv[i] * fv[j];
    }
  }
  // Spectra deduces a member function: Matrix operator* (const Eigen::Ref< const Matrix > &mat_in) const
};

// Represents the matrix L = (D1 * D1.T) where D1 := vertex/edge oriented boundary matrix
template< typename F = float, bool lex_order = true >
struct UpLaplacian1_lowerstar {
  using Scalar = F;  // required by Eigen 
  using VectorXS = Vector< Scalar >;
  const size_t nv;
  const size_t ne; 
  mutable vector< Scalar > r;  // need mutability to re-use workspace with const declared member functions
  const Vector< Scalar > fv;   // filter values (triplet form)
  const VectorXu I;            // row indices (triplet form)
  const VectorXu J;            // col indices (triplet form)
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
  res["n_restarts"] = solver.num_iterations();    // number of 'restarts' / Lanczos iterations; num times a new residual vector is generated and orthogonalized w.r.t V
  res["n_operations"] = solver.num_operations();  // number of matvec products 
  res["v0"] = v0;
  res["nev"] = nev; // number of requested eigenvalues
  res["ncv"] = ncv; // number of Lanczos vectors
  res["ritz_values"] = solver.ritz_values();
  res["ritz_converged"] = solver.ritz_converged();
  res["status"] = status; 
  return(res);
}

// Implicitly restarted Lanczos 
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
  UpLaplacian1_lowerstar< float > op(fv, I, J); // a, b, eps, w
  return IRL< UpLaplacian1_lowerstar< float > >(std::move(op), nev, ncv, max_iter, tol);
};


// max_iter := number of Lanczos iterations 
auto UpLaplacian0_VELS_IRL(VectorXf fv, VectorXu I, VectorXu J, const Index nev, const Index ncv, const Index max_iter, const float tol) -> py::dict {
  SmoothStep< float, false > se = { 0.0, 0.0, 0.0, 0.0 };
  SmoothStep< float, true >  sv = { 0.0, 0.0, 0.0, 0.0 };
  UpLaplacian0_VELS< float > op(fv, I, J, sv, se);
  return IRL< UpLaplacian0_VELS< float > >(std::move(op), nev, ncv, max_iter, tol);
};


// clang -Wall -fPIC -c src/pbsig/lanczos_spectra.cpp -std=c++17 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -Iextern/eigen -Iextern/spectra/include
PYBIND11_MODULE(_lanczos, m) {
  m.doc() = "Lanczos multiplication module";
  m.def("sparse_lanczos", &SparseMatIRL, "Implicitly restarted Lanczos on sparse matrix");
  m.def("UL1_LS_matvec", &UpLaplacian1_matvec, "Up-Laplacian1 lower-star matrix-vector multiplication");
  m.def("UL1_LS_lanczos", &UpLaplacian1_IRL, "Up-Laplacian1 lower-star Lanczos (implicit restart)");
  m.def("UL0_VELS_lanczos", &UpLaplacian0_VELS_IRL, "Up-Laplacian0 lower-star Lanczos (implicit restart)");
  // m.def("sparse_lanczos_init", &SparseLanczosInit, "Implicitly restarted Lanczos on sparse matrix");
  // m.def("UL1_LS_jacobi_davidson", &UpLaplacian1_JacobiDavidson, "A function that adds two numbers");
}