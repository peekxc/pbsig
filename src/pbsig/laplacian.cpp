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

// [[nodiscard]]
// inline auto lex_unrank_2_array(const uint_64 r, const size_t n) noexcept -> std::array< uint_64, 2 > {
//   auto i = static_cast< uint_64 >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
//   auto j = static_cast< uint_64 >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
//   return(std::array< uint_64, 2 >{ i, j });
// }

// template < typename I = int64_t, typename Hasher = pthash::murmurhash2_64, typename Encoder = pthash::dictionary_dictionary >
// struct IndexMap {
//   typedef I value_type;
//   typedef pthash::single_phf< Hasher, Encoder, true > pthash_type;   
  
//   pthash::build_configuration config;      // hash configuration
//   pthash_type pmhf;                        // pmhf
//   vector< I > offsets;                     // offsets to get the right index
  
//   // Index map
//   IndexMap(float c = 6.0, float alpha = 0.94, bool minimal = true, bool verbose = false){
//     config.c = c;
//     config.alpha = alpha;
//     config.minimal_output = minimal;  // makes perfect hash function *minimal*
//     config.verbose_output = verbose;
//   }

//   template< typename InputIt > 
//   void build(InputIt b, const InputIt e){
//     const size_t n_elems = std::distance(b,e);
//     pmhf.build_in_internal_memory(b, n_elems, config);
    
//     offsets.resize(n_elems);
//     I i = 0; 
//     std::for_each(b,e,[&](auto elem){
//       auto key = (I) pmhf(elem);
//       offsets.at(key) = i++;
//     });
//   };

//   // Not safe! must pass exact values here
//   [[nodiscard]]
//   constexpr auto operator[](I key) const noexcept -> I {
//     return offsets[pmhf(key)];
//   };
// };

// /* Compute and print the number of bits spent per key. */
// double bits_per_key = static_cast<double>(index_map_pmhf.num_bits()) / index_map_pmhf.num_keys();
// py::print("function uses ", bits_per_key, " [bits/key]");

// /* Sanity check! */
// if (check(keys.begin(), keys.size(), f)) std::cout << "EVERYTHING OK!" << std::endl;

// mutable unordered_map< I, I > index_map; 


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

// const vector< uint_64 > qr;             // p+1 ranks
// vector< uint_64 > pr;    
// nv;
//   const size_t np;
//   const size_t nq 
template< class Laplacian > 
auto _simplices_from_ranks(const Laplacian& L) -> py::array_t< int > {
  vector< int > simplices; 
  simplices.reserve(static_cast< int >(L.nq*(L.dim+2)));
  auto out = std::back_inserter(simplices);
  combinatorial::lex_unrank(L.qr.begin(), L.qr.end(), size_t(L.nv), size_t(L.dim+2), out);
  array< ssize_t, 2 > _shape = { static_cast< ssize_t >(L.nq), L.dim+2 };
  return py::array_t< int , py::array::c_style | py::array::forcecast >(_shape, simplices.data());
  // return py::cast(simplices);
}

template< class Laplacian > 
auto _faces_from_ranks(const Laplacian& L) -> py::array_t< int > {
  vector< int > faces; 
  const size_t np_actual = L.pr.size();
  faces.reserve(static_cast< int >(np_actual*(L.dim+1)));
  auto out = std::back_inserter(faces);
  combinatorial::lex_unrank(L.pr.begin(), L.pr.end(), size_t(L.nv), size_t(L.dim+1), out);
  array< ssize_t, 2 > _shape = { static_cast< ssize_t >(np_actual), L.dim+1 };
  return py::array_t< int , py::array::c_style | py::array::forcecast >(_shape, faces.data());
}


template< int p, typename F >
void declare_laplacian(py::module &m, std::string typestr) {
  using Class = UpLaplacian< p, F, true >;
  std::string pyclass_name = std::string("UpLaplacian") + typestr;
  using array_t_FF = py::array_t< F, py::array::f_style | py::array::forcecast >;
  py::class_< Class >(m, pyclass_name.c_str())
    .def(py::init< const vector< uint64_t >, size_t, size_t >())
    .def_readonly("shape", &Class::shape)
    .def_readonly("nv", &Class::nv)
    .def_readonly("np", &Class::np)
    .def_readonly("nq", &Class::nq)
    .def_readonly("qr", &Class::qr)
    .def_readonly("pr", &Class::pr)
    .def_readwrite("fpl", &Class::fpl)
    .def_readwrite("fpr", &Class::fpr)
    .def_readwrite("fq", &Class::fq)
    .def_readonly("degrees", &Class::degrees)
    .def_property_readonly("dtype", [](const Class& L){
      auto dtype = pybind11::dtype(pybind11::format_descriptor<F>::format());
      return dtype; 
    })
    .def_property_readonly("simplices", [](const Class& L){
      return _simplices_from_ranks(L);
    })
    .def_property_readonly("faces", [](const Class& L){
      return _faces_from_ranks(L);
    })
    .def("precompute_degree", &Class::precompute_degree)
    // .def("compute_indexes", &Class::compute_indexes)
    .def("_matvec", [](const Class& L, const py::array_t< F >& x) { return _matvec(L, x); })
    .def("_rmatvec", [](const Class& L, const py::array_t< F >& x) { return _matvec(L, x); })
    .def("_matmat", [](const Class& L, const array_t_FF& X){ return _matmat(L, X); })
    .def("_rmatmat", [](const Class& L, const array_t_FF& X){ return _matmat(L, X); })
    ;
}
auto boundary_ranks(const size_t p_rank, const size_t n, const size_t k) -> py::array_t< int > {
  std::vector< I > br(k,0);
  boundary(p_rank, n, k, br.begin());
  return py::cast(br);
}

// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/laplacian.cpp -std=c++17 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_laplacian, m) {
  m.doc() = "Laplacian multiplication module";
  declare_laplacian< 0, double >(m, "0D");
  declare_laplacian< 0, float >(m, "0F");
  declare_laplacian< 1, double >(m, "1D");
  declare_laplacian< 1, float >(m, "1F");
  declare_laplacian< 2, double >(m, "2D");
  declare_laplacian< 2, float >(m, "2F");
  m.def("decompress_faces", &decompress_faces, "Decompresses ranks");
  m.def("boundary_ranks", &boundary_ranks, "Gets boundary ranks from a given rank");
}

