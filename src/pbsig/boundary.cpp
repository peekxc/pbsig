// cppimport
#include <cstdint>
#include <array>
#include <cmath>
#include <bitset>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for implicit STL conversions 

namespace py = pybind11;

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

py::tuple lower_star_boundary_1(const py::array_t< double >& W, const double diam){
  auto w = W.unchecked< 1 >();
  vector< int > col_indices;
  vector< double > data; 
  vector< double > weights; 
  auto edge = std::array< sz_t, 2 >(); 
  const auto chain = (std::array< double, 2 >){ 1, -1 }; 
  
  sz_t N = static_cast< sz_t >(W.shape(0));
  for (sz_t i = 0; i < N; ++i){
    for (sz_t j = i+1; j < N; ++j){
      edge[0] = i; 
      edge[1] = j; 
      double iw = static_cast< double >(w(i));
      double jw = static_cast< double >(w(j));
      double ew = std::max(iw, jw);
      if (ew <= diam){
        col_indices.insert(col_indices.end(), edge.begin(), edge.end());
        data.insert(data.end(), chain.begin(), chain.end());
        weights.push_back(ew);
      }
    }
  }
  
  const int ne = int(data.size()/2);
  auto offsets = vector< int >();
  offsets.reserve(ne + 1);
  for (int i = 0; i < (ne + 1); ++i){ offsets.push_back(2*i); }

  py::array_t< int > CI(col_indices.size(), col_indices.data());
  py::array_t< double > CD(data.size(), data.data());
  py::array_t< int > CO(offsets.size(), offsets.data());
  py::array_t< double > WD(weights.size(), weights.data());
  return(py::make_tuple(CD, CI, CO, WD));
}

py::tuple rips_boundary_1(const py::array_t< double >& D, const size_t n, const double diam) {
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

// Constructs a edge-triangle boundary matrix 
// Precondition: edge ranks must be sorted lexicographically!
// Given vertex weights 'W', edge ranks 'ER' and triangle ranks 'TR', produces the indices needed to construct a CSC sparse matrix 
// Note that the potentially large edge and triangle ranks are copied on passing. The implicit conversion is the simplest way to do this. 
#include <cstdint> 
py::tuple lower_star_boundary_2_sparse(const py::array_t< double >& W, const std::vector< uint64_t > ER, const std::vector< uint64_t > TR) {
  if (!std::is_sorted(ER.begin(), ER.end())){  return(py::make_tuple(-1)); }
  //if (!std::is_sorted(TR.begin(), TR.end())){  return(py::make_tuple(-1)); }
  const sz_t nv = static_cast< sz_t >(W.shape(0));
  const sz_t nt = TR.size(); 
  const auto w = W.unchecked< 1 >();
  vector< int > row_indices;
  vector< double > data; 
  vector< double > weights; 

  row_indices.reserve(3*nt);
  data.reserve(3*nt);
  weights.reserve(nt);

  auto CT = std::array< sz_t, 3 >();  // current triangle
  auto CTR = std::array< sz_t, 3 >(); // current triangle ranks
  std::array< double, 3 > chain = { 1, -1, 1 }; 
  for (sz_t j = 0; j < nt; ++j){
    unrank_lex_k(TR[j], 3, nv, static_cast< sz_t* >(CT.data()));
    CTR = { rank_lex_2(CT[0],CT[1],nv), rank_lex_2(CT[0],CT[2],nv), rank_lex_2(CT[1],CT[2],nv) };
    for (auto face_rank : CTR){
      auto it = std::lower_bound(ER.begin(), ER.end(), face_rank);
      row_indices.push_back(std::distance(ER.begin(), it));
    }
    data.insert(data.end(), chain.begin(), chain.end());
    weights.push_back(std::max(w(CT[0]), std::max(w(CT[1]), w(CT[2]))));
  }

  // For CSC matrices only 
  const int N = int(data.size()/3);
  auto offsets = vector< int >();
  offsets.reserve(N + 1);
  for (int i = 0; i < (N + 1); ++i){ offsets.push_back(3*i); }

  py::array_t< int > CI(row_indices.size(), row_indices.data());
  py::array_t< double > CD(data.size(), data.data());
  py::array_t< int > CO(offsets.size(), offsets.data());
  py::array_t< double > CW(weights.size(), weights.data());
  return(py::make_tuple(CD, CI, CO, CW)); // CSC matrix: data, row indices, column offsets, triangle weights
}

py::tuple lower_star_boundary_1_sparse(const py::array_t< double >& W, const std::vector< uint64_t > ER) {
  const sz_t nv = static_cast< sz_t >(W.shape(0));
  const sz_t ne = ER.size(); 
  const auto w = W.unchecked< 1 >();
  vector< int > row_indices;
  vector< double > data; 
  vector< double > weights; 

  row_indices.reserve(2*ne);
  data.reserve(2*ne);
  weights.reserve(ne);

  auto CE = std::array< sz_t, 2 >();  // current edge
  std::array< double, 2 > chain = { 1, -1 }; 
  for (sz_t j = 0; j < ne; ++j){
    unrank_lex_2(ER[j], nv, static_cast< sz_t* >(CE.data()));
    row_indices.insert(row_indices.end(), CE.begin(), CE.end());
    data.insert(data.end(), chain.begin(), chain.end());
    weights.push_back(std::max(w(CE[0]), w(CE[1])));
  }

  // For CSC matrices only 
  const int N = int(data.size()/2);
  auto offsets = vector< int >();
  offsets.reserve(N + 1);
  for (int i = 0; i < (N + 1); ++i){ offsets.push_back(2*i); }

  py::array_t< int > CI(row_indices.size(), row_indices.data());
  py::array_t< double > CD(data.size(), data.data());
  py::array_t< int > CO(offsets.size(), offsets.data());
  py::array_t< double > CW(weights.size(), weights.data());
  return(py::make_tuple(CD, CI, CO, CW)); // CSC matrix: data, row indices, column offsets, triangle weights
}

py::tuple lower_star_boundary_2(const py::array_t< double >& W, const double diam) {
  const sz_t n = static_cast< sz_t >(W.shape(0));
  const auto w = W.unchecked< 1 >();
  vector< int > col_indices;
  vector< double > data; 
  vector< double > weights; 
  auto T = std::array< sz_t, 3 >(); 
  std::array< double, 3 > chain = { 1, -1, 1 }; 
  for (sz_t i = 0; i < n; ++i){
    for (sz_t j = i+1; j < n; ++j){
      for (sz_t k = j+1; k < n; ++k){
        T = { rank_lex_2(i,j,n), rank_lex_2(i,k,n), rank_lex_2(j,k,n) };
        double tw = std::max(w(i), std::max(w(j), w(k)));
        if (tw <= diam){
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
  py::array_t< double > CW(weights.size(), weights.data());
  return(py::make_tuple(CD, CI, CO, CW)); // matrix data, column indices, row offsets, triangle weights
}

py::tuple rips_boundary_2(const py::array_t< double >& D, const size_t n, const double diam) {
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

// For naming convention, see: https://github.com/pybind/pybind11/issues/1004
PYBIND11_MODULE(_boundary, m) {
  m.doc() = "_boundary";
  m.def("rips_boundary_1", &rips_boundary_1);
  m.def("rips_boundary_2", &rips_boundary_2);
  m.def("lower_star_boundary_1", &lower_star_boundary_1);
  m.def("lower_star_boundary_2", &lower_star_boundary_2);
  m.def("lower_star_boundary_1_sparse", &lower_star_boundary_1_sparse);
  m.def("lower_star_boundary_2_sparse", &lower_star_boundary_2_sparse);
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
*/