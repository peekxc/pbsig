#include <array>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>
#include "combinatorial.h"

typedef float value_t;
typedef int64_t index_t;
using std::vector; 
using combinatorial::unrank_colex;
using combinatorial::unrank_lex;

template< bool colex = false, uint8_t dim > 
struct RankRange {
  const size_t n; 
  const vector< uint_fast64_t > ranks;

  RankRange(const size_t _n, const vector< uint_fast64_t >& _ranks) : n(_n), ranks(_ranks){}

  struct RankLabelIterator {
    const size_t _n;
    std::array< uint16_t, dim + 1 > labels;
    vector< uint_fast64_t >::const_iterator _it; 

    RankLabelIterator(const size_t __n, vector< uint_fast64_t >::const_iterator it) : _n(__n), _it(it){};
    
    const uint16_t* operator*() const {
      if constexpr (colex){
        unrank_colex(_it, _it+dim+1, _n, dim+1, (uint16_t*) &labels[0]);
      } else {
        unrank_lex(_it, _it+dim+1, _n, dim+1, (uint16_t*) &labels[0]); 
      }
      return labels.data();
    }
    void operator++() {
      ++_it;
    }
    bool operator==(RankLabelIterator o) const {
      return _it == o._it;
    }
    bool operator!=(RankLabelIterator o) const {
      return _it != o._it;
    }
  };

  auto begin(){
    return RankLabelIterator(n, ranks.begin());
  }
  auto end(){
    return RankLabelIterator(n, ranks.end());
  }
};



struct binomial_coeff_table {
  std::vector<std::vector<index_t>> B;
  binomial_coeff_table(index_t n, index_t k) : B(k + 1, std::vector<index_t>(n + 1, 0)) {
    for (index_t i = 0; i <= n; ++i) {
      B[0][i] = 1;
      for (index_t j = 1; j < std::min(i, k + 1); ++j)
        B[j][i] = B[j - 1][i - 1] + B[j][i - 1];
      if (i <= k) B[i][i] = 1;
      //check_overflow(B[std::min(i >> 1, k)][i]);
    }
  }

  index_t operator()(index_t n, index_t k) const {
    assert(k < B.size() && n < B[k].size()  && n >= k - 1);
    return B[k][n];
  }
};

template <class Predicate>
index_t get_max(index_t top, const index_t bottom, const Predicate pred) {
  if (!pred(top)) {
    index_t count = top - bottom;
    while (count > 0) {
      index_t step = count >> 1, mid = top - step;
      if (!pred(mid)) {
        top = mid - 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }
  }
  return top;
}

index_t get_max_vertex(const index_t idx, const index_t k, const index_t n, const binomial_coeff_table& B) {
  return get_max(n, k - 1, [&](index_t w) -> bool { return (B(w, k) <= idx); });
}

template <typename OutputIterator>
OutputIterator get_simplex_vertices(
    index_t idx, 
    const index_t dim, 
    index_t n, 
    OutputIterator out, 
    const binomial_coeff_table& B
) {
  --n;
  for (index_t k = dim + 1; k > 1; --k) {
    n = get_max_vertex(idx, k, n, B);
    *out++ = n;
    idx -= B(n, k);
  }
  *out = idx;
  return out;
}

struct simplex_boundary_enumerator {
    index_t idx_below, idx_above, j, k;
    index_t dim;
    const binomial_coeff_table& B;

  public:
    simplex_boundary_enumerator(const index_t i, const index_t _dim, const index_t n, const binomial_coeff_table& _bt)
        : idx_below(i), idx_above(0), j(n - 1), k(_dim), B(_bt){}

    simplex_boundary_enumerator(const index_t _dim, const index_t n, const binomial_coeff_table& _bt)
        : simplex_boundary_enumerator(-1, _dim, n, _bt) {}

    void set_simplex(const index_t i, const index_t _dim, const index_t n) {
      idx_below = i;
      idx_above = 0;
      j = n - 1;
      k = _dim;
      dim = _dim;
    }

    bool has_next() { return (k >= 0); }

    index_t next() {
      j = get_max_vertex(idx_below, k + 1, j, B);
      index_t face_index = idx_above - B(j, k + 1) + idx_below;
      idx_below -= B(j, k + 1);
      idx_above += B(j, k);
      --k;
      return face_index;
    }
  };