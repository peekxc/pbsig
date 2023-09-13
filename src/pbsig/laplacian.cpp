#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "laplacian.h"
// #include "pthash.hpp"
// #include <omp.h>

// Namespace directives and declarations
namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal
using namespace combinatorial;

// Type aliases + alias templates 
using std::function; 
using std::vector;
using std::array;
using std::unordered_map;
using uint_32 = uint_fast32_t;
using uint_64 = uint_fast64_t;

// Matvec operation: Lx |-> y for any vector x
template< typename Laplacian, typename F = typename Laplacian::value_type > 
auto _matvec(const Laplacian& L, const py::array_t< F >& x_) noexcept -> py::array_t< F > {
  py::buffer_info x_buffer = x_.request();    // Obtain direct access
  L.__matvec(static_cast< F* >(x_buffer.ptr));  // stores result in internal y 
  return py::cast(L.y);
}

// Y = L @ X 
template< typename Laplacian, typename F = typename Laplacian::value_type > 
auto _matmat(const Laplacian& L, const py::array_t< F, py::array::f_style | py::array::forcecast >& X_) -> py::array_t< F > {
  const ssize_t n_rows = X_.shape(0);
  const ssize_t n_cols = X_.shape(1);
  // Obtain direct access
  py::buffer_info x_buffer = X_.request();
  F* X = static_cast< F* >(x_buffer.ptr);

  // Allocate memory 
  auto result = vector< F >();
  result.reserve(L.shape[0]*n_cols);

  // Each matvec outputs to y. Copy to result via output iterator
  auto out = std::back_inserter(result);
  for (ssize_t j = 0; j < n_cols; ++j){
    L.__matvec(X+(j*n_rows));
    std::copy(L.y.begin(), L.y.end(), out);
  }
  
  // From: https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h 
  array< ssize_t, 2 > Y_shape = { static_cast< ssize_t >(L.shape[0]), n_cols };
  return py::array_t< F , py::array::f_style | py::array::forcecast >(Y_shape, result.data());
}

template< class Laplacian > 
auto _simplices(const Laplacian& L) -> py::array_t< int > {
  vector< int > simplices; 
  simplices.reserve(static_cast< int >(L.nq*(L.dim+2)));
  for (auto s: L.simplices){
    for (size_t d = 0; d < L.dim+2; ++d){
      simplices.push_back(*(s+d));
    }
  }
  array< ssize_t, 2 > _shape = { static_cast< ssize_t >(L.nq), L.dim+2 };
  return py::array_t< int , py::array::c_style | py::array::forcecast >(_shape, simplices.data());
}

template< class Laplacian > 
auto _faces(const Laplacian& L) -> py::array_t< uint16_t > {
  auto face_ranks = unique_face_ranks(L.simplices, false);
  vector< uint16_t > faces; 
  auto out = std::back_inserter(faces);
  combinatorial::unrank_combs< Laplacian::colex_order >(face_ranks.begin(), face_ranks.end(), L.nv, L.dim+1, out);
  array< ssize_t, 2 > _shape = { static_cast< ssize_t >(L.np), L.dim+1 };
  return py::array_t< uint16_t , py::array::c_style | py::array::forcecast >(_shape, faces.data());
}


// autodiff include
// #include <autodiff/forward/real.hpp>
// #include <autodiff/forward/real/eigen.hpp>
// using namespace autodiff;
// using Eigen::MatrixXd;
// using Eigen::VectorXreal;

// // The vector function for which the Jacobian is needed
// template< typename Laplacian, typename F = typename Laplacian::value_type > 
// auto __lap_matvec(const VectorXreal& x, const VectorXreal& deg, const VectorXreal& fq, const VectorXreal& fp, const vector< int >& face_indices, size_t np, size_t nq) -> VectorXreal {  
//   VectorXreal y = VectorXreal(np);
//   std::transform(deg.begin(), deg.end(), x, y.begin(), std::multiplies< F >());
//   for (size_t qi = 0; qi < nq; ++qi){
//     const auto ii = face_indices[qi*3], jj = face_indices[qi*3+1], kk = face_indices[qi*3+2];
//     y[ii] += x[kk] * fp[ii] * fq[qi] * fp[kk] - x[jj] * fp[ii] * fq[qi] * fp[jj];
//     y[kk] += x[ii] * fp[kk] * fq[qi] * fp[ii] - x[jj] * fp[kk] * fq[qi] * fp[jj]; 
//     y[jj] -= x[ii] * fp[jj] * fq[qi] * fp[ii] + x[kk] * fp[jj] * fq[qi] * fp[kk]; 
//   }
//   return y;
// }

// // Matvec operation: Lx |-> y for any vector x
// template< typename Laplacian, typename F = typename Laplacian::value_type > 
// auto __matvecgrad(const Laplacian& L, const VectorXreal& x) -> MatrixXd {
//   VectorXreal y;
//   MatrixXd J = jacobian(__lap_matvec, wrt(x), at(x), y, L.degrees, L.fq, L.fpl, L.face_indices, L.np, L.nq);
//   return J;
// }

template< int p, typename F >
void declare_laplacian(py::module &m, std::string typestr, bool colex = false) {
  using Class = UpLaplacian< p, F, SimplexRange< p+1, false > >;

  std::string pyclass_name = std::string("UpLaplacian") + typestr;
  using array_t_FF = py::array_t< F, py::array::f_style | py::array::forcecast >;
  py::class_< Class >(m, pyclass_name.c_str())
    // .def(py::init< const vector< uint16_t >, size_t, size_t >())
    .def(py::init([](vector< uint16_t > simplices, const size_t n, const size_t np) {
      combinatorial::sort_contiguous(simplices, p+2, std::less< uint16_t>());
      auto rng = SimplexRange< p+1, false >(simplices, n);
      combinatorial::keep_table_alive = true; // needed for fast binomial coefficients
      return std::unique_ptr< Class >(new Class(rng, n, np));
    }))
    .def_readwrite("shape", &Class::shape)
    .def_readonly("nv", &Class::nv)
    .def_readonly("np", &Class::np)
    .def_readonly("nq", &Class::nq)
    .def_readwrite("fpl", &Class::fpl)
    .def_readwrite("fpr", &Class::fpr)
    .def_readwrite("fq", &Class::fq)
    .def_readonly("degrees", &Class::degrees)
    .def_property_readonly("dtype", []([[maybe_unused]] const Class& L){
      auto dtype = pybind11::dtype(pybind11::format_descriptor< typename Class::value_type >::format());
      return dtype; 
    })
    .def_property_readonly("simplices", [](const Class& L){
      return _simplices(L);
    })
    .def_property_readonly("faces", [](const Class& L){
      return _faces(L);
    })
    .def("precompute_degree", &Class::precompute_degree)
    .def("precompute_indices", &Class::precompute_indices)
    // .def("apply_mask", &Class::apply_mask)
    // .def("compute_indexes", &Class::compute_indexes)
    .def("_matvec", [](const Class& L, const py::array_t< F >& x) { return _matvec(L, x); })
    .def("_rmatvec", [](const Class& L, const py::array_t< F >& x) { return _matvec(L, x); })
    .def("_matmat", [](const Class& L, const array_t_FF& X){ return _matmat(L, X); })
    .def("_rmatmat", [](const Class& L, const array_t_FF& X){ return _matmat(L, X); })
    // .def("__matvecgrad", [](const Class& L, const VectorXreal& X){ return __matvecgrad(L, X); })
    ;
}

// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/laplacian.cpp -std=c++20 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -I/Users/mpiekenbrock/pbsig/extern/eigen -I/Users/mpiekenbrock/pbsig/include
PYBIND11_MODULE(_laplacian, m) {
  m.doc() = "Laplacian multiplication module";
  declare_laplacian< 0, double >(m, "0D");
  declare_laplacian< 0, float >(m, "0F");
  declare_laplacian< 1, double >(m, "1D");
  declare_laplacian< 1, float >(m, "1F");
  declare_laplacian< 2, double >(m, "2D");
  declare_laplacian< 2, float >(m, "2F");
  // m.def("decompress_faces", &decompress_faces, "Decompresses ranks");
  // m.def("boundary_ranks", &boundary_ranks, "Gets boundary ranks from a given rank");
}

