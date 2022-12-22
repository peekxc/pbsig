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
using std::vector;
using uint_32 = uint_fast32_t;
using uint_64 = uint_fast64_t;

[[nodiscard]]
inline auto lex_unrank_2_array(const uint_64 r, const size_t n) noexcept -> std::array< uint_64, 2 > {
  auto i = static_cast< uint_64 >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
  auto j = static_cast< uint_64 >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
  return(std::array< uint_64, 2 >{ i, j });
}

template< typename F = float, bool lex_order = true >
struct UpLaplacian0_LS {
  const size_t nv;
  const size_t ne; 
  std::array< size_t, 2 > shape; 
  const vector< uint64_t > er; // edge ranks
  mutable vector< F > y;  // workspace
  vector< F > fv;         // filter values 
  vector< F > Df;         // weighted degrees; compute during initialization
  UpLaplacian0_LS(const vector< uint64_t > er_, const size_t nv_) : nv(nv_), ne(er_.size()), er(er_), y(nv_)  {
    shape = { nv, nv };
  }

  // Precomputes the degree term
  void precompute_degree(){
    if (fv.size() != nv || er.size() != ne){ return; }
    Df = vector< F >(nv);
    F ew = 0.0;
    for (size_t cc = 0; cc < ne; ++cc){
      auto [i, j] = lex_unrank_2_array(er[cc], nv);
      ew = std::max(fv[i], fv[j]);
      Df[i] += ew * ew;
      Df[j] += ew * ew;
    }
  };

  // Matvec operation: Lx |-> y for any vector x
  auto _matvec(const py::array_t< F >& x) const -> py::array_t< F > {
    // Ensure workplace vectors are zero'ed
    // auto xe = x.unchecked< float, 1 >();
    py::buffer_info x_buffer = x.request();
    F* xe = static_cast< F* >(x_buffer.ptr);

    std::fill(y.begin(), y.end(), 0);
    for (size_t cc = 0; cc < nv; ++cc){
      y[cc] += xe[cc] * Df[cc] * fv[cc] * fv[cc];
    }
    F ew; 
    size_t i, j;
    for (size_t cc = 0; cc < ne; ++cc){
      auto [i, j] = lex_unrank_2_array(er[cc], nv);
      ew = std::max(fv[i], fv[j]);
      y[i] -= xe[j] * ew * ew * fv[i] * fv[j];
      y[j] -= xe[i] * ew * ew * fv[i] * fv[j];
    }
    return py::cast(y);
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