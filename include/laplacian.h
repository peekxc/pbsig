// Header includes 
#include <cinttypes>
#include <cstdint>
#include <array>
#include <span>
#include <cmath>	 // round, sqrt, floor
#include <numeric> // midpoint, accumulate
#include <unordered_map> 
#include <concepts>
#include <array>
#include <iterator>
#include <ranges>
#include <span> 
#include "combinatorial.h"
#include "pmh.h"
#include "splex_ranges.h"
#include <ranges>
#include <unordered_set>
#include <omp.h>


using namespace combinatorial;

// Type aliases + alias templates 
using std::function; 
using std::vector;
using std::array;
using std::unordered_map;
using uint_32 = uint_fast32_t;
using uint_64 = uint_fast64_t;

template< typename T >
concept SimplexIterable = requires(T a){
  { a.begin().operator*() } -> std::convertible_to< uint16_t* >;
  // { a.boundary() }
}; 
// && std::ranges::range< T >; 
// && std::forward_iterator< T >;

//static_assert(std::forward_iterator<RankLabelIterator>);

// Given codimension-1 ranks, determines the ranks of the corresponding faces
// p := simplex dimension of given ranks 
// auto decompress_faces(const vector< uint_64 >& cr, const size_t n, const size_t p, bool unique) -> vector< uint_64 > {
//   vector< uint_64 > fr; 
//   fr.reserve(cr.size()); 
//   for (auto ci : cr){
//     combinatorial::apply_boundary(ci, n, p+1, [&](auto face_rank){ fr.push_back(face_rank); });
//   }
//   if (unique) {
//     std::sort(fr.begin(), fr.end());
//     fr.erase(std::unique(fr.begin(), fr.end()), fr.end());
//   }
//   return fr;
// };


// Unique boundary face ranks; optionally order-preserving
// NOTE: By default order-preserving is off; NEED colex or lex ordering to ensure sign pattern is constant
template< SimplexIterable S_Iter > 
auto unique_face_ranks(S_Iter& simplices, bool preserve_order = false) -> vector< uint_64 > {
  std::unordered_set< uint_64 > seen;
  vector< uint64_t > face_ranks; 
  face_ranks.reserve(std::distance(simplices.begin(), simplices.end())); 
  for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it){
    s_it.boundary_ranks([&](auto face_rank){
      if (seen.insert(face_rank).second) {
        face_ranks.push_back(face_rank);
      }
    });
  }
  if (!preserve_order){ std::sort(face_ranks.begin(), face_ranks.end()); }
  // face_ranks.erase(std::unique(face_ranks.begin(), face_ranks.end()), face_ranks.end());
  return face_ranks;
};

template< int p = 0, typename F = double, SimplexIterable S_Iterable = SimplexRange< p+1, true >, IntegralHashTable H = std::unordered_map< uint64_t, uint32_t >  >
struct UpLaplacian {
  static constexpr int dim = p;           // Laplacian dimension
  static constexpr bool colex_order = S_Iterable::colex_order; // whether to use colex ordering for boundary bijection 
  using value_type = F;                   // field coefficient type
  const size_t nv;                        // number of vertices
  const size_t nq;                        // number of cofaces
  size_t np;                              // number of faces
  array< size_t, 2 > shape;               // non-const due due to masking 
  S_Iterable simplices;                   // q simplex range
  mutable vector< F > y;                  // workspace
  mutable H index_map;                    // indexing function
  vector< F > fpl;                        // p-simplex left weights 
  vector< F > fpr;                        // p-simplex right weights 
  vector< F > fq;                         // (p+1)-simplex weights
  vector< F > degrees;                    // weighted degrees; pre-computed
  vector< uint32_t > face_indices;        // precomputed face indices  

  UpLaplacian(S_Iterable S, const size_t nv_, const size_t np_ = 0) // np is needed for p-faces having no codim-1 faces! 
    : nv(nv_), nq(std::distance(S.begin(), S.end())), simplices(S)  {
    auto pr = unique_face_ranks(simplices, false); // NEED colex or lex ordering to ensure sign pattern is constant
    np = std::max(pr.size(), np_);
    shape = { np, np };
    y = vector< F >(np); // todo: experiment with local _alloca allocation
    fpl = vector< F >(np, 1.0);
    fpr = vector< F >(np, 1.0); // not necessary if symmetric 
    fq = vector< F >(nq, 1.0); 
    degrees = vector< F >(np, 0.0);

    // Build the Hash table mappping face ranks -> indices
    vector< std::pair< uint64_t, uint32_t > > key_values; 
    key_values.reserve(pr.size());
    for (uint64_t i = 0; i < pr.size(); ++i){
      key_values.push_back(std::make_pair(pr[i], i));
    }
    index_map = H(key_values.begin(), key_values.end());
    precompute_indices();
  }

  void precompute_indices(){
    if constexpr(p == 0){
      face_indices.clear();
      for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it){
        const auto [i,j] = s_it.boundary_ranks();
        const auto ii = index_map[i], jj = index_map[j];
        face_indices.push_back(ii);
        face_indices.push_back(jj);
      }
    }
    else if constexpr(p == 1){
      face_indices.clear();
      face_indices.reserve(3*nq);
      for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it){
        const auto [i,j,k] = s_it.boundary_ranks();
        const auto ii = index_map[i], jj = index_map[j], kk = index_map[k];
        face_indices.push_back(ii);
        face_indices.push_back(jj);
        face_indices.push_back(kk);
      }
    } 
    // else if constexpr(p == 2) {
    //   face_indices.clear();
    //   face_indices.reserve(4*nq);
    //   for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it){
    //     const auto [i,j,k,l] = s_it.boundary_ranks();
    //     const auto ii = index_map[i], jj = index_map[j], kk = index_map[k], ll = index_map[l];
    //     face_indices.push_back(ii);
    //     face_indices.push_back(jj);
    //     face_indices.push_back(kk);
    //     face_indices.push_back(ll);
    //   }
    else {
      size_t cc = 0; 
      face_indices.clear();
      face_indices.reserve((dim+2)*nq);
      auto p_ind = array< uint64_t, dim+2 >();
      for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it){
        cc = 0; 
        s_it.boundary_ranks([this, &cc, &p_ind](auto face_rank){ p_ind[cc++] = index_map[face_rank]; });
        face_indices.insert(face_indices.end(), p_ind.begin(), p_ind.end()); 
      }
    }
  }

  void precompute_degree(){
    if (fpl.size() != np || fq.size() != nq || fpl.size() != np){ return; }
    std::fill_n(degrees.begin(), degrees.size(), 0);
    size_t qi = 0; 
    for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it, ++qi){
      // std::cout << "qi: " << qi << std::endl;
      s_it.boundary_ranks([&](auto face_rank){ 
        const auto ii = index_map[face_rank]; 
        degrees.at(ii) += fpl.at(ii) * fq.at(qi) * fpr.at(ii);
      });
    }
  }

  // Internal matvec; outputs y = L @ x
  inline void __matvec(F* x) const noexcept { 
    // Start with the degree computation
    std::transform(degrees.begin(), degrees.end(), x, y.begin(), std::multiplies< F >());

    if constexpr(p == 0){
      // #pragma omp parallel for schedule (static,16)
      for (size_t qi = 0; qi < nq; ++qi){
        const auto ii = face_indices[qi*2], jj = face_indices[qi*2+1];
        y[ii] -= x[jj] * fpl[ii] * fq[qi] * fpr[jj]; 
        y[jj] -= x[ii] * fpl[jj] * fq[qi] * fpr[ii];
      }
    } else if constexpr (p == 1){
      // #pragma omp parallel for schedule (static,16)
      // #pragma omp simd // NOTE: dont do this, it wont work without re-working the computation
      for (size_t qi = 0; qi < nq; ++qi){
        const auto ii = face_indices[qi*3], jj = face_indices[qi*3+1], kk = face_indices[qi*3+2];
        y[ii] += x[kk] * fpl[ii] * fq[qi] * fpr[kk] - x[jj] * fpl[ii] * fq[qi] * fpr[jj];
        y[kk] += x[ii] * fpl[kk] * fq[qi] * fpr[ii] - x[jj] * fpl[kk] * fq[qi] * fpr[jj]; 
        y[jj] -= x[ii] * fpl[jj] * fq[qi] * fpr[ii] + x[kk] * fpl[jj] * fq[qi] * fpr[kk]; 
      }
    }
    
    // #pragma omp parallel for simd schedule (static,16)
    // size_t qi = 0; 
    // for (auto s_it = simplices.begin(); s_it != simplices.end(); ++s_it, ++qi){
    //   if constexpr(p == 0){
    //     const auto [i,j] = s_it.boundary_ranks();
    //     const auto ii = index_map[i], jj = index_map[j];
    //     // const auto ii = face_indices[cc*2], jj = face_indices[cc*2+1];
    //     y[ii] -= x[jj] * fpl[ii] * fq[qi] * fpr[jj]; 
    //     y[jj] -= x[ii] * fpl[jj] * fq[qi] * fpr[ii];
    //   } else if constexpr(p == 1){
    //     const auto [i,j,k] = s_it.boundary_ranks();
    //     const auto ii = index_map[i], jj = index_map[j], kk = index_map[k];
        // // const auto ii = face_indices[cc*3], jj = face_indices[cc*3+1], kk = face_indices[cc*3+2];
        // // std::cout << ii << ", " << jj << ", " << kk << std::endl;
        // y[ii] -= x[jj] * fpl[ii] * fq[qi] * fpr[jj];
        // y[jj] -= x[ii] * fpl[jj] * fq[qi] * fpr[ii]; 
        // y[ii] += x[kk] * fpl[ii] * fq[qi] * fpr[kk]; 
        // y[kk] += x[ii] * fpl[kk] * fq[qi] * fpr[ii]; 
        // y[jj] -= x[kk] * fpl[jj] * fq[qi] * fpr[kk]; 
        // y[kk] -= x[jj] * fpl[kk] * fq[qi] * fpr[jj]; 
    //   } else {
    //     auto p_ind = array< uint64_t, dim+2 >();
    //     int s = 1; 
    //     s_it.boundary_ranks([this, &cc, &p_ind](auto face_rank){ p_ind[cc++] = index_map[face_rank]; });
    //     for_each_combination(begin(p_ind), begin(p_ind)+2, end(p_ind), [&](auto a, auto b){
    //       const auto ii = *a;
    //       const auto jj = *(b-1); // to remove compiler warning
    //       y[ii] += s * x[jj] * fpl[ii] * fq[qi] * fpr[jj]; 
    //       y[jj] += s * x[ii] * fpl[jj] * fq[qi] * fpr[ii];
    //       s = -s; 
    //       return false; 
    //     });
    //   }
    // } // end simplices loop 
  } // __matvec


}; // UpLaplacian

