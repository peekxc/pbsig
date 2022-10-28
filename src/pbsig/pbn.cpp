// Header includes 
#include <cinttypes>
#include <numeric> // iota 
#include <tuple> // make_tuple, tie 
#include <algorithm>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <pybind11/iostream.h>
namespace py = pybind11;

using std::vector;
using std::tie; 

#include "disjoint_set.h"

// ## Compute lower-star edge values; prepare to traverse in order
// # fe = fv[E].max(axis=1)  # function evaluated on edges
// ne = E.shape[0]
// ei = np.fromiter(sorted(range(ne), key=lambda i: (max(fv[E[i,:]]), min(fv[E[i,:]]))), dtype=int)

template< typename T > 
inline auto cast_vector(py::array_t< T >& a) noexcept -> std::vector< T > {
  py::buffer_info buf = a.request();  
  const T* p = static_cast< T* >(buf.ptr);
  return std::vector< T >(size_t(buf.size), *p);
}

// Assumes edges come in sorted order 
void ph0_lower_star(py::array_t< double > fv, py::array_t< int > EI, py::array_t< int > EJ){
  auto f = cast_vector< double >(fv);
  auto U = cast_vector< int >(EI);
  auto V = cast_vector< int >(EJ);

  // Data structures + variables
  const size_t nv = f.size();
  const size_t ne = U.size();
  auto ds = DisjointSet< unsigned int > (nv);

  // Elder map to maintain minimum set representatives ("elders")
  auto elders = vector< int >(nv);
  std::iota(elders.begin(), elders.end(), 0);

  // Whether vertex i is paired 
  auto paired = vector< bool >(nv, false);
  
  // Proceed to union components via elder rule
  double fi;
  int elder, child, creator, u, v; 
  for (size_t i = 0; i < ne; ++i){
    fi = f[i];
    u = U[i], v = V[i];
    tie(elder, child) = f[u] <= f[v] ? std::make_tuple(u, v) : std::make_tuple(v, u);
    if (!ds.connected(u,v)){
      if (!paired[child]){
        paired[child] = true;
      } else {
        creator = f[elders[ds[child]]] <= f[elders[ds[elder]]] ? elders[ds[elder]] : elders[ds[child]];
        paired[creator] = true; 
      }
    }
    elders[ds[u]] = elders[ds[v]] = elders[ds[u]] <= elders[ds[v]] ? elders[ds[u]] : elders[ds[v]];
    ds.merge(u,v)
  }
}

// $ c++ -O2 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) pbn.cpp -o pbn$(python3-config --extension-suffix)
PYBIND11_MODULE(_pbn, m) {
  m.doc() = "PBN module";
  m.def("ph0_lower_star", &ph0_lower_star, "Computes 0-d persistence diagram of a lower star filtration");
}

  // dgm = []
  // for (i,j), f in zip(E[ei,:], fv[E[ei,:]].max(axis=1)):
  //   ## The 'elder' was born first before the child: has smaller function value
  //   # assert i != 7 and j != 8 # should kill 5! 
  //   elder, child = (i, j) if fv[i] <= fv[j] else (j, i)
  //   # if child == 5 and elder == 6: assert False
  //   if not ds.connected(i,j):
  //     if not paired[child]: # child unpaired => merged instantly by (i,j)
  //       dgm += [(fv[child], f)] if insert_rule(fv[child], f) else []
  //       paired[child] = True # kill the child
  //       #assert child != 5
  //       #print(f"{child}, ({i},{j})")
  //     else: # child already paired in component, use elder rule (keep elder alive)
  //       #assert elder != 8 # problem child
  //       creator = elders[ds[elder]] if fv[elders[ds[child]]] <= fv[elders[ds[elder]]] else elders[ds[child]]
  //       dgm += [(fv[creator], f)] if insert_rule(fv[creator], f) else []
  //       # assert not(paired[creator])
  //       paired[creator] = True
  //       #print(f"{creator}, ({i},{j})")
      
  //   ## Merge (i,j) and update elder map
  //   elder_i, elder_j = elders[ds[i]], elders[ds[j]]
  //   ds.merge(i,j)
  //   elders[ds[i]] = elders[ds[j]] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j



 
