// Header includes 
#include <cinttypes>
#include <cstdint>
#include <array>
#include <span>
#include <cmath>	 // round, sqrt, floor
#include <numeric> // midpoint, accumulate
#include <unordered_map> 

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "combinatorial.h"
#include "pthash.hpp"

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

[[nodiscard]]
inline auto lex_unrank_2_array(const uint_64 r, const size_t n) noexcept -> std::array< uint_64, 2 > {
  auto i = static_cast< uint_64 >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
  auto j = static_cast< uint_64 >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
  return(std::array< uint_64, 2 >{ i, j });
}


template < typename I = int64_t, typename Hasher = pthash::murmurhash2_64, typename Encoder = pthash::dictionary_dictionary >
struct IndexMap {
  typedef I value_type;
  typedef pthash::single_phf< Hasher, Encoder, true > pthash_type;   
  
  pthash::build_configuration config;      // hash configuration
  pthash_type pmhf;                        // pmhf
  vector< I > offsets;                     // offsets to get the right index
  
  // Index map
  IndexMap(float c = 6.0, float alpha = 0.94, bool minimal = true, bool verbose = true){
    config.c = c;
    config.alpha = alpha;
    config.minimal_output = minimal;  // makes perfect hash function *minimal*
    config.verbose_output = verbose;
  }

  template< typename InputIt > 
  void build(InputIt b, const InputIt e){
    const size_t n_elems = std::distance(b,e);
    pmhf.build_in_internal_memory(b, n_elems, config);
    
    offsets.resize(n_elems);
    I i = 0; 
    std::for_each(b,e,[&](auto elem){
      auto key = (I) pmhf(elem);
      offsets.at(key) = i++;
    });
  };

  // Not safe! must pass exact values here
  constexpr auto operator[](I key) const noexcept -> I {
    return offsets[pmhf(key)];
  };
};

// /* Compute and print the number of bits spent per key. */
// double bits_per_key = static_cast<double>(index_map_pmhf.num_bits()) / index_map_pmhf.num_keys();
// py::print("function uses ", bits_per_key, " [bits/key]");

// /* Sanity check! */
// if (check(keys.begin(), keys.size(), f)) std::cout << "EVERYTHING OK!" << std::endl;

// mutable unordered_map< I, I > index_map; 

// TODO: remove type-erased std::function binding via templates for added performance
template< int p = 0, typename F = double, bool lex_order = true >
struct UpLaplacian {
  using Map_t = IndexMap< uint_64, pthash::murmurhash2_64, pthash::dictionary_dictionary >;
  // using Map_t = unordered_map< uint_64, uint_64 >;
  const size_t nv;
  const size_t np;
  const size_t nq; 
  array< size_t, 2 > shape;  // TODO: figure out how to initialize in constructor
  const vector< uint_64 > qr;            // p+1 ranks
  // const function< uint_32 (uint64_t) > h; // indexing function 
  mutable vector< F > y;                  // workspace
  // mutable unordered_map< uint_64, uint_64 > index_map;    // indexing function ; mutabel needed for dumb reasons
  Map_t index_map; 
  vector< F > fpl;                        // p-simplex left weights 
  vector< F > fpr;                        // p-simplex right weights 
  vector< F > fq;                         // (p+1)-simplex weights
  vector< F > degrees;                    // weighted degrees; pre-computed

  UpLaplacian(const vector< uint_64 > qr_, const size_t nv_, const size_t np_) 
    : nv(nv_), np(np_), nq(qr_.size()), qr(qr_)  {
    shape = { np, np };
    y = vector< F >(np); // todo: experiment with local _alloca allocation
    fpl = vector< F >(np, 1.0);
    fpr = vector< F >(np, 1.0);
    fq = vector< F >(nq, 1.0); 
    degrees = vector< F >(np, 0.0);
  }

  // Prepares indexing hash function 
  void compute_indexes(){
    //auto index_map = std::unordered_map< uint64_t, uint64_t >; // Maps face ranks to sequential indices 
    vector< uint64_t > fr;
    fr.reserve(qr.size()); 

    // Collect the faces 
    for (auto qi : qr){
      combinatorial::apply_boundary(qi, nv, p+2, [&](auto face_rank){ fr.push_back(face_rank); });
    }

    // Only consider unique faces; sort by lexicographical ordering
    std::sort(fr.begin(), fr.end());
    fr.erase(std::unique(fr.begin(), fr.end()), fr.end());

    // Build the index map
    index_map.build(fr.begin(), fr.end());
    // auto indices = vector< uint64_t >(fr.size(), 0);
    // std::iota(indices.begin(), indices.end(), 0);

    // // Create the index map
    // for (uint64_t i = 0; i < fr.size(); ++i){
    //   index_map.emplace(fr[i], i);
    // }
        // auto it = index_map.find(fr);
        // if (it == index_map.end()) {
        //   index_map.emplace_hint(it, fr, cc);
        //   cc++;
        // }
  }

  // Precomputes the degree term
  void precompute_degree(){
    if (fpl.size() != np || fq.size() != nq || fpl.size() != np){ return; }
    std::fill_n(degrees.begin(), degrees.size(), 0);
    size_t q_ind = 0; 
    for (auto qi : qr){
      combinatorial::apply_boundary(qi, nv, p+2, [&](auto face_rank){ 
        const auto ii = index_map[face_rank];
        degrees.at(ii) += fpl.at(ii) * fq.at(q_ind) * fpr.at(ii);
      });
      q_ind += 1; 
    }
  };

  // Internal matvec; outputs y = L @ x
  inline void __matvec(F* x) const noexcept { 
    // Start with the degree computation
    std::transform(degrees.begin(), degrees.end(), x, y.begin(), std::multiplies< F >());

    // The matvec
    size_t q_ind = 0;
    auto p_ranks = array< uint64_t, p+2 >();

    #pragma omp simd
    for (auto qi: qr){
      if constexpr (p == 0){
				lex_unrank_2(static_cast< I >(qi), static_cast< I >(nv), begin(p_ranks));
        const auto ii = p_ranks[0]; // index_map[q_vertices[0]]; 
        const auto jj = p_ranks[1];
        y[ii] -= x[jj] * fpl[ii] * fq[q_ind] * fpr[jj]; 
        y[jj] -= x[ii] * fpl[jj] * fq[q_ind] * fpr[ii]; 
      } else if constexpr (p == 1) { // inject the sign @ compile time
        boundary(qi, nv, p+2, begin(p_ranks));
        const auto ii = index_map[p_ranks[0]];
        const auto jj = index_map[p_ranks[1]];
        const auto kk = index_map[p_ranks[2]];
        y[ii] -= x[jj] * fpl[ii] * fq[q_ind] * fpr[jj];
        y[jj] -= x[ii] * fpl[jj] * fq[q_ind] * fpr[ii]; 
        y[ii] += x[kk] * fpl[ii] * fq[q_ind] * fpr[kk]; 
        y[kk] += x[ii] * fpl[kk] * fq[q_ind] * fpr[ii]; 
        y[jj] -= x[kk] * fpl[jj] * fq[q_ind] * fpr[kk]; 
        y[kk] -= x[jj] * fpl[kk] * fq[q_ind] * fpr[jj]; 
      } else {
        auto p_ranks = array< uint64_t, p+2 >();
        boundary(qi, nv, p+2, begin(p_ranks));
        size_t cc = 0;
        const array< float, 2 > sgn_pattern = { -1.0, 1.0 };
				for_each_combination(begin(p_ranks), begin(p_ranks)+2, end(p_ranks), [&](auto a, auto b){
          const auto ii = index_map[*a];
          const auto jj = index_map[*(b-1)]; // to remove compiler warning
					y[ii] += sgn_pattern[cc] * x[jj] * fpl[ii] * fq[q_ind] * fpr[jj]; 
          y[jj] += sgn_pattern[cc] * x[ii] * fpl[jj] * fq[q_ind] * fpr[ii];
          cc = (cc + 1) % 2;
          return false; 
        });
      }
      q_ind += 1;
    }
  }

  // Matvec operation: Lx |-> y for any vector x
  auto _matvec(const py::array_t< F >& x_) const noexcept -> py::array_t< F > {
    py::buffer_info x_buffer = x_.request();    // Obtain direct access
    __matvec(static_cast< F* >(x_buffer.ptr));  // stores result in internal y 
    return py::cast(y);
  }

  // Y = L @ X 
  auto _matmat(const py::array_t< F, py::array::f_style | py::array::forcecast >& X_) const -> py::array_t< F > {
    const ssize_t n_rows = X_.shape(0);
    const ssize_t n_cols = X_.shape(1);
    // Obtain direct access
    py::buffer_info x_buffer = X_.request();
    F* X = static_cast< F* >(x_buffer.ptr);

    // Allocate memory 
    auto result = vector< F >();
    result.reserve(shape[0]*n_cols);

    // Each matvec outputs to y. Copy to result via output iterator
    auto out = std::back_inserter(result);
    for (ssize_t j = 0; j < n_cols; ++j){
      __matvec(X+(j*n_rows));
      std::copy(y.begin(), y.end(), out);
    }
   
    // From: https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h 
    array< ssize_t, 2 > Y_shape = { static_cast< ssize_t >(shape[0]), n_cols };
    return py::array_t< F , py::array::f_style | py::array::forcecast >(Y_shape, result.data());
  } 

  //  x = x.reshape(-1)
  //   self._v.fill(0)
  //   self._v += self.degree * x
  //   for s_ind, s in enumerate(self.simplices):
  //     for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
  //       ii, jj = self.index(f1), self.index(f2)
  //       self._v[ii] += sgn_ij * x[jj] * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
  //       self._v[jj] += sgn_ij * x[ii] * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
  //   return self._v
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

// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/laplacian.cpp -std=c++17 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_laplacian, m) {
  m.doc() = "Laplacian multiplication module";
  py::class_< UpLaplacian< 0, float, true > >(m, "UpLaplacian0")
    .def(py::init< const vector< uint64_t >, size_t, size_t >())
    .def_readonly("shape", &UpLaplacian< 0, float, true >::shape)
    .def_readonly("nv", &UpLaplacian< 0, float, true >::nv)
    .def_readonly("np", &UpLaplacian< 0, float, true >::np)
    .def_readonly("nq", &UpLaplacian< 0, float, true >::nq)
    .def_readonly("qr", &UpLaplacian< 0, float, true >::qr)
    .def_readwrite("fpl", &UpLaplacian< 0, float, true >::fpl)
    .def_readwrite("fpr", &UpLaplacian< 0, float, true >::fpr)
    .def_readwrite("fq", &UpLaplacian< 0, float, true >::fq)
    // .def_readonly("index_map", &UpLaplacian< 0, float, true >::index_map)
    .def_readonly("degrees", &UpLaplacian< 0, float, true >::degrees)
    .def("precompute_degree", &UpLaplacian< 0, float, true >::precompute_degree)
    .def("compute_indexes", &UpLaplacian< 0, float, true >::compute_indexes)
    .def("_matvec", &UpLaplacian< 0, float, true >::_matvec)
    .def("_matmat", &UpLaplacian< 0, float, true >::_matmat);
    // .def("benchmark_index", &UpLaplacian< 0, float, true >::benchmark_index);
  py::class_< UpLaplacian< 1, float, true > >(m, "UpLaplacian1")
    .def(py::init< const vector< uint64_t >, size_t, size_t >())
    .def_readonly("shape", &UpLaplacian< 1, float, true >::shape)
    .def_readonly("nv", &UpLaplacian< 1, float, true >::nv)
    .def_readonly("np", &UpLaplacian< 1, float, true >::np)
    .def_readonly("nq", &UpLaplacian< 1, float, true >::nq)
    .def_readonly("qr", &UpLaplacian< 1, float, true >::qr)
    .def_readwrite("fpl", &UpLaplacian< 1, float, true >::fpl)
    .def_readwrite("fpr", &UpLaplacian< 1, float, true >::fpr)
    .def_readwrite("fq", &UpLaplacian< 1, float, true >::fq)
    // .def_readonly("index_map", &UpLaplacian< 1, float, true >::index_map)
    .def_readonly("degrees", &UpLaplacian< 1, float, true >::degrees)
    .def("precompute_degree", &UpLaplacian< 1, float, true >::precompute_degree)
    .def("compute_indexes", &UpLaplacian< 1, float, true >::compute_indexes)
    .def("_matvec", &UpLaplacian< 1, float, true >::_matvec)
    .def("_matmat", &UpLaplacian< 1, float, true >::_matmat);
    // .def("benchmark_index", &UpLaplacian< 1, float, true >::benchmark_index);
    
    // .def("_matvec", &UpLaplacian0_LS< float, true >::_matvec)
    // .def("precompute_degree", &UpLaplacian0_LS< float, true >::precompute_degree);
  // m.def("vectorized_func", py::vectorize(my_func));s
  //m.def("call_go", &call_go);
}

  // const size_t nv;
  // const size_t np;
  // const size_t nq; 
  // array< size_t, 2 > shape;  // TODO: figure out how to initialize in constructor
  // const vector< uint64_t > qr;            // p+1 ranks
  // const function< uint_32 (uint64_t) > h; // indexing function 
  // mutable vector< F > y;                  // workspace
  // vector< F > fpl;                        // p-simplex left weights 
  // vector< F > fpr;                        // p-simplex right weights 
  // vector< F > fq;                         // (p+1)-simplex weights
  // vector< F > Df;       