// Header includes 
#include <cinttypes>
#include <cstdint>
#include <array>
#include <span>
#include <cmath>	 // round, sqrt, floor
#include <numeric> // midpoint, accumulate

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Namespace directives and declarations
namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

// Type aliases + alias templates 
using std::function; 
using std::vector;
using std::array;
using uint_32 = uint_fast32_t;
using uint_64 = uint_fast64_t;

[[nodiscard]]
inline auto lex_unrank_2_array(const uint_64 r, const size_t n) noexcept -> std::array< uint_64, 2 > {
  auto i = static_cast< uint_64 >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
  auto j = static_cast< uint_64 >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
  return(std::array< uint_64, 2 >{ i, j });
}

// TODO: remove type-erased std::function binding via templates for added performance
template< int p = 0, typename F = float, bool lex_order = true >
struct UpLaplacian {
  const size_t nv;
  const size_t np;
  const size_t nq; 
  array< size_t, 2 > shape;  // TODO: figure out how to initialize in constructor
  const vector< uint64_t > qr;            // p+1 ranks
  const function< uint_32 (uint64_t) > h; // indexing function 
  mutable vector< F > y;                  // workspace
  vector< F > fpl;                        // p-simplex left weights 
  vector< F > fpr;                        // p-simplex right weights 
  vector< F > fq;                         // (p+1)-simplex weights
  vector< F > Df;                         // weighted degrees; pre-computed
  UpLaplacian(const vector< uint64_t > qr_, const size_t nv_, const size_t np_) 
    : nv(nv_), np(np_), nq(qr_.size()), qr(qr_)  {
    shape = { nv, nv };
    y = vector< F >(np); // todo: experiment with local _alloca allocation
    fpl = vector< F >(np, 1.0);
    fpr = vector< F >(np, 1.0);
    fq = vector< F >(nq, 1.0); 
    Df = vector< F >(np, 0.0);
    prepare();
  }

  // Precomputes the degree term
  // boundary ranks
  void prepare(){
    if (fp.size() != np || fq.size() != nq || Df.size() != np){ return; }
    // F qw = 0.0;
    // for (size_t qi = 0; qi < nq; ++qi){
    //   std::array< size_t, p+1 > q_simplex = lex_unrank_2_array(qr[qi], nv);
    //   for (size_t qi = 0; qi < nq; ++qi){
    //   std::fill(y.begin(), y.end(), 0);
    //     for (size_t cc = 0; cc < nv; ++cc){
    //       y[cc] += xe[cc] * Df[cc] * fv[cc] * fv[cc];
    //     }
    //   }
    // }
  };

  // Matvec operation: Lx |-> y for any vector x
  auto _matvec(const py::array_t< F >& x) const -> py::array_t< F > {
    // Ensure workplace vectors are zero'ed
    // auto xe = x.unchecked< float, 1 >();
    // py::buffer_info x_buffer = x.request();
    // F* xe = static_cast< F* >(x_buffer.ptr);
    // F ew; 
    // size_t i, j;
    // std::array< size_t, p+1 > q_simplex = lex_unrank_2_array(qr[qi], nv);
    // size_t cc = 0;
    // combinations(q_simplex, 2, [&cc](std::array< size_t, p >& pi, std::array< size_t, p >& pj){
    //   i = h(lex_rank(pi, nv));
    //   j = h(lex_rank(pj, nv));
    //   std::pow(-1, cc % 2)*;
    // });
    // return py::cast(y);
    return py::cast(x);
  }
};

// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/laplacian.cpp -std=c++17 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_laplacian, m) {
  m.doc() = "Laplacian multiplication module";
  py::class_< UpLaplacian0_LS< float, true > >(m, "UpLaplacian0_LS")
    .def(py::init< const vector< uint64_t >, size_t >())
    .def_readwrite("fv", &UpLaplacian0_LS< float, true >::fv)
    .def("_matvec", &UpLaplacian0_LS< float, true >::_matvec)
    .def("precompute_degree", &UpLaplacian0_LS< float, true >::precompute_degree);
  // m.def("vectorized_func", py::vectorize(my_func));s
  //m.def("call_go", &call_go);
}