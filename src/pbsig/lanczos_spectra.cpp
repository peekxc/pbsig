#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <Spectra/SymEigsSolver.h>

#include <cinttypes>
using namespace Spectra;

using std::vector;
using uint_32 = uint_fast32_t;
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
  // vector< uint_32 > I; 
  // vector< uint_32 > J;
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

auto UpLaplacian1_Lanczos(VectorXf fv, VectorXu I, VectorXu J, int nev, int ncv) -> Eigen::VectorXf {
  UpLaplacian1_lowerstar< true > op(fv, I, J);
  
  // https://spectralib.org/doc/classspectra_1_1symeigssolver
  SymEigsSolver< UpLaplacian1_lowerstar< true > > eigs(op, nev, ncv); // nev, ncv
  eigs.init();
  eigs.compute(SortRule::LargestAlge);
  if(eigs.info() == CompInfo::Successful){
    Eigen::VectorXf evalues = eigs.eigenvalues();
    return(evalues); 
  }
  Eigen::VectorXf Z(fv.size());
  return(Z);
}

// clang -Wall -fPIC -c src/pbsig/lanczos_spectra.cpp -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -Iextern/eigen -Iextern/spectra/include
PYBIND11_MODULE(_lanczos, m) {
  m.doc() = "Lanczos multiplication module";
  m.def("UL1_LS_matvec", &UpLaplacian1_matvec, "A function that adds two numbers");
  m.def("UL1_LS_lanczos", &UpLaplacian1_Lanczos, "A function that adds two numbers");
}