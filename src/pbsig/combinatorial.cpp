#include "combinatorial.h"
#include <cinttypes>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
// #include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iterator>     // std::back_inserter
#include <vector>       // std::vector
#include <algorithm>    // std::copy
using std::vector; 

auto rank_combs(py::array_t< uint16_t > combs, const int n, const int k, bool colex = true) -> py::array_t< uint64_t > {
  py::buffer_info buffer = combs.request();
  uint16_t* p = static_cast< uint16_t* >(buffer.ptr);
  const size_t N = buffer.size;
  vector< uint64_t > ranks; 
  ranks.reserve(static_cast< uint64_t >(N/k));
  auto out = std::back_inserter(ranks);
	if (colex) {
		combinatorial::rank_lex(p, p+N, size_t(n), size_t(k), out);
	} else {
		combinatorial::rank_colex(p, p+N, size_t(n), size_t(k), out);
	}
  return py::cast(ranks); 
}

// auto unrank_combs(py::array_t< int > ranks, const int n, const int k) -> py::array_t< int > {
//   py::buffer_info buffer = ranks.request();
//   int* r = static_cast< int* >(buffer.ptr);
//   const size_t N = buffer.size;
//   vector< int > simplices; 
//   simplices.reserve(static_cast< int >(N*k));
//   auto out = std::back_inserter(simplices);
//   combinatorial::unrank_lex(r, r+N, size_t(n), size_t(k), out);
//   return py::cast(simplices);
// }

// auto boundary_ranks(const int p_rank, const int n, const int k) -> py::array_t< int > {
//   vector< int > face_ranks = vector< int >();
// 	combinatorial::apply_boundary(p_rank, n, k, [&face_ranks](size_t r){
//     face_ranks.push_back(r);
//   });
//   return py::cast(face_ranks);
// }


// Given interval [s1, s2] and a set of other intervals O = { [o1,o2], [o3,o4], ..., [ok,ol] }, 
// computes net change in length moving s1 to position s2 would have on all intervals in O
template< typename Iter >
int interval_cost(Iter s, Iter ob, Iter oe) {
  const int s1 = s[0];
	const int s2 = s[1];
	int sum = 0;
	if (s1 == s2){ return(0); }
	if (s1 < s2){
		for (auto o_it = ob; o_it != oe; o_it += 2){
			const int o1 = *o_it;
			const int o2 = *(o_it+1);
			if (o1 == o2){ continue; }
			else if (o1 < o2){
				if (s1 < o1){
					sum += ((s2 < o1 || s2 >= o2) ? 0 : 1);
				} else if (s1 > o1 && s1 < o2) { 
					sum += s2 < o2 ? 0 : -1;
				} 
			} else {
				if (s2 == o2){ sum += 1; } 
				else if (s2 > o2){ sum += s2 > o1 ? 0 : (s2 == o1 ? -1 : 1); } 
			}
		}
	} else {// s1 > s2 
		for (auto o_it = ob; o_it != oe; o_it += 2){
			const int o1 = *o_it;
			const int o2 = *(o_it+1);
			if (o1 == o2){ continue; }
			else if (o1 < o2){
				if (s1 > o1){ sum += s1 <= o2 ? -1 : 0; } 
			} else { 
				if (s1 > o2){ sum += s1 < o1 ? -1 : 0; }
			}
		}
	}
	return(sum);
}

// Given interval [s1, s2] and a set of other intervals O = { [o1,o2], [o3,o4], ..., [ok,ol] }, computes 
// net change in length moving s1 to position s2 would have on all intervals in O
// int interval_cost_rcpp(std::vector< int > s, std::vector< int > O) {
// 	return(interval_cost(s.begin(), O.begin(), O.end()));
// }

// Given a set of directed intervals in contiguous memory [i1,i2,j1,j2,...,l1,l2], returns the interval cost 
// of displacing each interval with respect to the rest of the intervals in the set
auto pairwise_cost(std::vector< int > indices) -> py::array_t< int >{
	const size_t n = static_cast< size_t >(indices.size()/2); 
	std::vector< int > res(n, 0);
	for (size_t k = 0; k < n; ++k){
		res[k] = interval_cost(indices.begin(), indices.begin() + 2, indices.end());
		
    // swap the memory to go through the other intervals
    if (k < (n - 1)){
			std::swap(indices[0], indices[2*(k+1)]);
			std::swap(indices[1], indices[2*(k+1)+1]);
		}
	}
	return(py::cast(res));
}

// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/combinatorial.cpp -std=c++20 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 
PYBIND11_MODULE(_combinatorial, m) {
  m.doc() = "Combinatorial module";
  m.def("rank_combs", &rank_combs);
  // m.def("unrank_combs", &unrank_combs);
  // m.def("boundary_ranks", &boundary_ranks);
  m.def("interval_cost", &pairwise_cost);
  // m.def("vectorized_func", py::vectorize(my_func));s
  //m.def("call_go", &call_go);
}