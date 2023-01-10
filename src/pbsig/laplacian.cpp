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

// TODO: remove type-erased std::function binding via templates for added performance
template< int p = 0, typename F = double, bool lex_order = true >
struct UpLaplacian {
  const size_t nv;
  const size_t np;
  const size_t nq; 
  array< size_t, 2 > shape;  // TODO: figure out how to initialize in constructor
  const vector< uint_64 > qr;            // p+1 ranks
  // const function< uint_32 (uint64_t) > h; // indexing function 
  mutable vector< F > y;                  // workspace
  mutable unordered_map< uint_64, uint_64 > index_map;    // indexing function ; mutabel needed for dumb reasons
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

    // Create the index map
    for (uint64_t i = 0; i < fr.size(); ++i){
      index_map.emplace(fr[i], i);
    }
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

  // Matvec operation: Lx |-> y for any vector x
  auto _matvec(const py::array_t< F >& x_) const -> py::array_t< F > {
    // Ensure workplace vectors are zero'ed
    //std::fill_n(y.begin(), y.size(), static_cast< F >(0.0));
    
    // Obtain direct access
    py::buffer_info x_buffer = x_.request();
    F* x = static_cast< F* >(x_buffer.ptr);

    // Start with the degree computation
    std::transform(degrees.begin(), degrees.end(), x, y.begin(), std::multiplies< F >());

    // The matvec
    size_t q_ind = 0;
    auto q_vertices = array< uint_64, p+2 >();
    for (auto qi: qr){
      if constexpr (p == 0){
				lex_unrank_2(static_cast< I >(qi), static_cast< I >(nv), begin(q_vertices));
        const auto ii = index_map[q_vertices[0]]; // TODO: could speed up by assuming vertices start from 0 / remove index map
        const auto jj = index_map[q_vertices[1]];// TODO: could speed up by assuming vertices start from 0 / remove index map
        y[ii] -= x[jj] * fpl[ii] * fq[q_ind] * fpr[jj]; // inject the sign @ compile time
        y[jj] -= x[ii] * fpl[jj] * fq[q_ind] * fpr[ii]; // inject the sign @ compile time
      } else {
				// lex_unrank_k(static_cast< I >(qi), static_cast< I >(nv), p+2, begin(q_vertices));
        auto p_ranks = array< uint64_t, p+2 >();
        size_t cc = 0;
        apply_boundary(qi, nv, p+2, [&](auto face_rank){
          p_ranks[cc] = static_cast< uint64_t >(face_rank); // todo: output it + transform since same size! 
          cc++;
        });
        const array< float, 2 > sgn_pattern = { -1.0, 1.0 };
        cc = 0;
				for_each_combination(begin(p_ranks), begin(p_ranks)+2, end(p_ranks), [&](auto a, auto b){
          const auto ii = index_map[*a];
          const auto jj = index_map[*(a+1)];
					y[ii] += sgn_pattern[cc] * x[jj] * fpl[ii] * fq[q_ind] * fpr[jj]; // inject the sign @ compile time
          y[jj] += sgn_pattern[cc] * x[ii] * fpl[jj] * fq[q_ind] * fpr[ii]; // inject the sign @ compile time
          cc = (cc + 1) % 2;
          return false; 
        });
      }
      q_ind += 1;
    }
    return py::cast(y);
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
    .def_readonly("index_map", &UpLaplacian< 0, float, true >::index_map)
    .def_readonly("degrees", &UpLaplacian< 0, float, true >::degrees)
    .def("precompute_degree", &UpLaplacian< 0, float, true >::precompute_degree)
    .def("compute_indexes", &UpLaplacian< 0, float, true >::compute_indexes)
    .def("_matvec", &UpLaplacian< 0, float, true >::_matvec);
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
    .def_readonly("index_map", &UpLaplacian< 1, float, true >::index_map)
    .def_readonly("degrees", &UpLaplacian< 1, float, true >::degrees)
    .def("precompute_degree", &UpLaplacian< 1, float, true >::precompute_degree)
    .def("compute_indexes", &UpLaplacian< 1, float, true >::compute_indexes)
    .def("_matvec", &UpLaplacian< 1, float, true >::_matvec);
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