#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/DavidsonSymEigsSolver.h>

#include <cinttypes>
using namespace Spectra;

using std::vector;
using uint_32 = uint_fast32_t;
using VectorXd = Eigen::VectorXd;
using VectorXf = Eigen::VectorXf;
using VectorXu = Eigen::Matrix< uint_32, Eigen::Dynamic, 1 >;

// Represents the matrix L = (D1 * D1.T) where D1 := vertex/edge oriented boundary matrix
template< bool lex_order = true >
struct UpLaplacian1_lowerstar {
  using Scalar = float;  // required
  
  const size_t nv;
  const size_t ne; 
  mutable vector< Scalar > r;
  mutable vector< Scalar > rz;
  VectorXf fv;
  VectorXu I;
  VectorXu J; 
  UpLaplacian1_lowerstar(VectorXf fv_, VectorXu I_, VectorXu J_) : nv(fv_.size()), ne(I_.size()), fv(fv_), I(I_), J(J_) {
    r = vector< Scalar >(ne, 0.0);
    rz = vector< Scalar >(nv, 0.0);
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

auto UpLaplacian1_matvec(VectorXf x, VectorXf fv, VectorXu I, VectorXu J) -> Eigen::VectorXf {
  UpLaplacian1_lowerstar< true > op(fv, I, J);
  VectorXf y = VectorXf::Zero(fv.size());
  op.perform_op(x.data(), y.data());
  return(y);
}

using namespace pybind11::literals;  // to bring in the `_a` literal
using Index = Eigen::Index;



auto UpLaplacian1_Lanczos(VectorXf fv, VectorXu I, VectorXu J, int nev, int ncv) -> py::dict {
  UpLaplacian1_lowerstar< true > op(fv, I, J);
  SymEigsSolver< UpLaplacian1_lowerstar< true > > eigs(op, nev, ncv); // nev, ncv
  
  // Solve for eigenvalues and sort by largest magnitude
  eigs.init();
  eigs.compute(SortRule::LargestMagn);
  
  // If successful, return converged results 
  py::dict res;
  if(eigs.info() == CompInfo::Successful){
    Eigen::VectorXf evalues = eigs.eigenvalues();
    // res["n_restarts"]=eigs.n_restarts();
    // res["n_operations"]=eigs.n_operations();
    // res["eigenvalues"]=evalues;
    return(res);
  }
  return(res);
  // auto solver_base = SymEigsBase< UpLaplacian1_lowerstar< true >, IdentityBOp >(op, IdentityBOp(), nev, ncv);
  
  // const Index n_ops = solver_base.m_nmatop; // number of matrix operations called
  // const Index n_restarts = solver_base.m_niter; // number of matrix operations called
  // // LanczosFac    m_fac;        // Lanczos factorization
  // // Vector        m_ritz_val;   // Ritz values

  // const VectorXf evals = solver_base.eigenvalues();
  // Matrix        m_ritz_vec;   // Ritz vectors
  // Vector        m_ritz_est;   // last row of m_ritz_vec, also called the Ritz estimates
  // BoolArray     m_ritz_conv;  // indicator of the convergence of Ritz values
  // CompInfo      m_info;       // Successful, NotComputed,  NotConverging, NumericalIssue  
};

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/Util/SimpleRandom.h>


auto SparseLanczosInit(Eigen::SparseMatrix< double > M, const Index nev, const Index ncv, const Index max_iter, const double tol, VectorXd v0) -> py::dict { // 
  SparseSymMatProd< double > op(M);
  SymEigsSolver< SparseSymMatProd<double> > eigs(op, nev, ncv);
  eigs.init_v0(v0);
  int nconv = eigs.compute_ev(SortRule::LargestMagn, max_iter, tol, SortRule::LargestAlge); // SortRule, maxit, tol, sorting
  Eigen::VectorXd evalues;
  std::string status = ""; 
  if (eigs.info() == CompInfo::Successful){
    evalues = eigs.eigenvalues();
    status = "success";
  } else if (eigs.info() == CompInfo::NotConverging){
    status = "not converged";
  } else if (eigs.info() == CompInfo::NumericalIssue){
    status = "numerical issues encountered";
  } else {
    status = "failed";
  }
  py::dict res;
  res["eigenvalues"] = evalues;
  res["n_restarts"] = eigs.num_iterations();
  res["n_operations"] = eigs.num_operations();
  res["v0"] = v0;
  res["nev"] = nev;
  res["ncv"] = ncv;
  res["ritz_values"] = eigs.ritz_values();
  res["ritz_converged"] = eigs.ritz_converged();
  res["status"] = status; 
  return(res);
}

auto SparseLanczos(Eigen::SparseMatrix< double > M, const Index nev, const Index ncv, const Index max_iter, const double tol) -> py::dict { 
  SimpleRandom< double > rng(0);
  VectorXd v0 = rng.random_vec(M.rows());
  return SparseLanczosInit(M, nev, ncv, max_iter, tol, v0);
}
  // https://spectralib.org/doc/classspectra_1_1symeigssolver
  // SymEigsSolver< UpLaplacian1_lowerstar< true > > eigs(op, nev, ncv); // nev, ncv
  // eigs.init();
  // eigs.compute(SortRule::LargestMagn);
  // if(eigs.info() == CompInfo::Successful){
  //   Eigen::VectorXf evalues = eigs.eigenvalues();
  //   return(evalues); 
  // }
  
  
//   Eigen::VectorXf Z(fv.size());
//   return(Z);
// }


// auto UpLaplacian1_JacobiDavidson(VectorXf fv, VectorXu I, VectorXu J, int nev, int nvec_init, int nvec_max) -> Eigen::VectorXf {
//   UpLaplacian1_lowerstar< true > op(fv, I, J);
  
//   // https://spectralib.org/doc/classspectra_1_1symeigssolver
//   Spectra::DavidsonSymEigsSolver< UpLaplacian1_lowerstar< true > > eigs(op, nev, nvec_init, nvec_max); // nev, ncv
//   eigs.init();
  
//   eigs.compute(SortRule::LargestMagn);
//   if(eigs.info() == CompInfo::Successful){
//     Eigen::VectorXf evalues = eigs.eigenvalues();
//     return(evalues); 
//   }
//   Eigen::VectorXf Z(fv.size());
//   return(Z);
// }

// clang -Wall -fPIC -c src/pbsig/lanczos_spectra.cpp -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -Iextern/eigen -Iextern/spectra/include
PYBIND11_MODULE(_lanczos, m) {
  m.doc() = "Lanczos multiplication module";
  m.def("UL1_LS_matvec", &UpLaplacian1_matvec, "Up-Laplacian1 lower-star matrix-vector multiplication");
  m.def("UL1_LS_lanczos", &UpLaplacian1_Lanczos, "Up-Laplacian1 lower-star Lanczos (implicit restart)");
  m.def("sparse_lanczos", &SparseLanczos, "Implicitly restarted Lanczos on sparse matrix");
  m.def("sparse_lanczos_init", &SparseLanczosInit, "Implicitly restarted Lanczos on sparse matrix");
  // m.def("UL1_LS_jacobi_davidson", &UpLaplacian1_JacobiDavidson, "A function that adds two numbers");
}