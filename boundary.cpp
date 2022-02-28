// cppimport
#include <cstdint>
#include <array>
#include <cmath>
#include <bitset>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// result = np.zeros(k, dtype=int)
//   x = 1
//   for i in range(1, k+1):
//     while(r >= comb(n-x, k-i)):
//       r -= comb(n-x, k-i)
//       x += 1
//     result[i-1] = (x - 1)
//     x += 1

using std::vector; 
using sz_t = std::uint64_t;

sz_t binomial_coefficient(const int n, const int k) {
  double res = 1;
  for (int i = 1; i <= k; ++i){
    res = res * (n - k + i) / i;
  }
  return (sz_t)(res + 0.01);
}

// Lexicographically unrank 2-subsets
inline void unrank_lex_2(const sz_t r, const sz_t n, sz_t* out) noexcept  {
	auto i = static_cast< sz_t >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
	auto j = static_cast< sz_t >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
	out[0] = i;
	out[1] = j; 
}

// Lexicographically unrank k-subsets
inline void unrank_lex_k(sz_t r, const size_t k, const size_t n, sz_t* out){
	size_t x = 1; 
	for (size_t i = 1; i <= k; ++i){
		while(r >= binomial_coefficient(n-x, k-i)){
			r -= binomial_coefficient(n-x, k-i);
			x += 1;
		}
		*out++ = (x - 1);
		x += 1;
	}
}

inline auto rank_lex_2(sz_t i, sz_t j, const sz_t n) noexcept -> sz_t {
  if (j < i){ std::swap(i,j); }
  return sz_t(n*i - i*(i+1)/2 + j - i - 1);
}

// Lexicographically rank k-subsets
template< typename InputIter > [[nodiscard]]
inline sz_t rank_lex_k(InputIter s, const size_t k, const size_t n, const sz_t N){
	sz_t i = k; 
  const sz_t index = std::accumulate(s, s+k, 0, [n, &i](sz_t val, sz_t num){ 
	  return val + binomial_coefficient((n-1) - num, i--); 
	});
  const sz_t combinadic = (N-1) - index; // Apply the dual index mapping
  return(combinadic);
}


py::tuple rips_boundary_matrix_1(const py::array_t< double >& D, const size_t n, const double diam) {
  auto d = D.unchecked< 1 >();
  vector< int > col_indices;
  vector< double > data; 
  vector< double > weights; 
  auto edge = std::array< sz_t, 2 >(); 
  const auto chain = (std::array< double, 2 >){ 1, -1 }; 
  sz_t N = static_cast< sz_t >(D.shape(0));
  for (sz_t c = 0; c < N; ++c){
    double ew = static_cast< double >(d(c));
    if (ew <= diam){
      unrank_lex_2(c, n, edge.data());
      col_indices.insert(col_indices.end(), edge.begin(), edge.end());
      data.insert(data.end(), chain.begin(), chain.end());
      weights.push_back(ew);
    }
  }
  
  const int ne = int(data.size()/2);
  auto offsets = vector< int >();
  offsets.reserve(ne + 1);
  for (int i = 0; i < (ne + 1); ++i){ offsets.push_back(2*i); }

  py::array_t< int > CI(col_indices.size(), col_indices.data());
  py::array_t< double > CD(data.size(), data.data());
  py::array_t< int > CO(offsets.size(), offsets.data());
  py::array_t< double > W(weights.size(), weights.data());
  return(py::make_tuple(CD, CI, CO, W));
}

py::tuple rips_boundary_matrix_2_dense(const py::array_t< double >& D, const size_t n, const double diam) {
  const auto d = D.unchecked< 1 >();
  vector< int > col_indices;
  vector< double > data; 
  vector< double > weights; 
  auto T = std::array< sz_t, 3 >(); 
  std::array< double, 3 > chain = { 1, -1, 1 }; 
  for (sz_t i = 0; i < n; ++i){
    for (sz_t j = i+1; j < n; ++j){
      for (sz_t k = j+1; k < n; ++k){
        T = { rank_lex_2(i,j,n), rank_lex_2(i,k,n), rank_lex_2(j,k,n) };
        // double ij_dist = d(T[0]), ik_dist = d(T[1]), jk_dist = d(T[2]);
        double tw = std::max(d(T[0]), std::max(d(T[1]), d(T[2])));
        if (tw <= diam){
          // py::print(T[0], ",", T[1], ",", T[2]);
          col_indices.insert(col_indices.end(), T.begin(), T.end());
          data.insert(data.end(), chain.begin(), chain.end());
          weights.push_back(tw);
        }
      }
    }
  }
  
  const int nt = int(data.size()/3);
  auto offsets = vector< int >();
  offsets.reserve(nt + 1);
  for (int i = 0; i < (nt + 1); ++i){ offsets.push_back(3*i); }

  py::array_t< int > CI(col_indices.size(), col_indices.data());
  py::array_t< double > CD(data.size(), data.data());
  py::array_t< int > CO(offsets.size(), offsets.data());
  py::array_t< double > W(weights.size(), weights.data());
  return(py::make_tuple(CD, CI, CO, W));
}

PYBIND11_MODULE(boundary, m) {
  m.def("rips_boundary_matrix_1", &rips_boundary_matrix_1);
  m.def("rips_boundary_matrix_2_dense", &rips_boundary_matrix_2_dense);
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
*/