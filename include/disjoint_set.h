//----------------------------------------------------------------------
//                        Disjoint-set data structure 
// File:                        disjoint_set.h
//----------------------------------------------------------------------
// Copyright (c) 2022 Matt Piekenbrock. All Rights Reserved.
//
// Class definition based off of data-structure described here:  
// https://en.wikipedia.org/wiki/Disjoint-set_data_structure

#include <cstdint>
#include <vector>
#include <algorithm>
#include <optional> // optional

using std::size_t;
using std::vector;
using std::optional;

// Fast and memory-unsafe disjoint set 
template< typename I = size_t > 
struct DisjointSet {
  size_t size; 
  vector< I > parent, rank; // rank <=> upper bound for its height
  DisjointSet(const size_t _size): size(_size), parent(_size), rank(_size, 0) {
    static_assert(std::is_unsigned< I >::value, "Type must be an insigned integral type.");
    std::iota(parent.begin(), parent.end(), 0);
  };
  
  // Find operation, using path compression
  auto _find(const I x) noexcept -> I {
    return parent[x] == x ? x : (parent[x] = _find(parent[x]));
  }
  auto Find(const I x) noexcept -> optional< I > {
    return (x >= size) ? std::nullopt : std::make_optional(_find(x));
  }
  
  // Convenience function 
  auto operator[](I&& x) noexcept -> I { return _find(std::forward<I>(x)); }

  // Convenience function 
  bool connected(const I x, const I y) noexcept {
    return _find(x) == _find(y);
  }

  // Union operation. Uses union-by-rank optimization to choose parent-subtrees
  // Returns a boolean indicating whether (x,y) were in separate components, or false if they were already in the same set.  
  bool _union(const I x, const I y) noexcept {
    const I xRoot = _find(x);
    const I yRoot = _find(y);
    if (xRoot == yRoot){ return false; }
    else if (rank[xRoot] > rank[yRoot]) { parent[yRoot] = xRoot; }
    else if (rank[xRoot] < rank[yRoot]) { parent[xRoot] = yRoot; }
    else if (rank[xRoot] == rank[yRoot]){
      parent[yRoot] = parent[xRoot];
      rank[xRoot] = rank[xRoot] + 1;
    }
    return true; 
  }

  // Safe Union 
  optional< bool > Union(const I x, const I y) noexcept {
    return (x >= size || y >= size) ? std::nullopt : std::make_optional(_union(x, y));    
  }

  // Add new sets 
  void AddSets(const size_t n_sets){
    parent.resize(size + n_sets);
    std::iota(parent.begin() + size, parent.end(), size); // parent initialized incrementally
    rank.resize(size + n_sets, 0); // rank all 0 
    size += n_sets; 
  }
  
  // Convenience functions
  template < class It >
  bool UnionAll(It b, const It e){
    if (std::distance(b, e) <= 1) { return false;  } // asserts d(b,e) >= 2 
    for (I i = *b, j = *(b+1); (b+1) != e; ++b){
      std::tie(i,j) = std::make_tuple(*b, *(b+1));
      Union(i,j);
    } 
  }
  // vector< size_t > FindAll(const vector< size_t >& idx){
  //   using idx_v = vector< size_t >;
  //   if (idx.size() == 0){ return idx_v(); }
  //   const size_t n = idx.size();
  //   idx_v cc = idx_v(n);
  //   std::transform(idx.begin(), idx.end(), cc.begin(), [=](const size_t i){
  //     return(Find(i));
  //   });
  //   return(cc);
  // }
}; // class DisjointSet