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
#include <concepts> 
#include <span> 
#include "combinatorial.h"


using namespace combinatorial;

// Type aliases + alias templates 
using std::function; 
using std::vector;
using std::array;
using std::unordered_map;
using uint_32 = uint_fast32_t;
using uint_64 = uint_fast64_t;


// template< typename T >
// concept LabelIterator = requires(T v){
//   { v.advance() }; 
//   { v.equals() };
//   { v.dereference() };
// };


// template < uint8_t d, std::input_or_output_iterator Iter >
// struct SimplexLabelIterator : public iterator_facade< SimplexLabelIterator<Iter>, std::contiguous_iterator<Iter> > {
//   using value_type = std::iter_value_t<Iter>;
//   using reference = std::iter_reference_t<Iter>;
//   using difference_type = std::iter_difference_t<Iter>;
//   template <std::input_or_output_iterator> friend class SimplexLabelIterator;
  
//   const std::vector< uint16_t > labels; // the full complex of simplices 
//   // std::array< uint16_t, d > cs;   // current simplex
//   Iter iter_;
    
//   constexpr SimplexLabelIterator(std::vector< uint16_t > _labels) noexcept(std::is_nothrow_constructible_v<Iter>) : labels(_labels) {};
//   constexpr SimplexLabelIterator(Iter iter) noexcept(std::is_nothrow_move_constructible_v<Iter>) : iter_(std::move(iter)) {}
//   template <class U>
//     requires(std::constructible_from<Iter, U> && !std::same_as<Iter, U>)
//     constexpr SimplexLabelIterator(SimplexLabelIterator<U> const& other) noexcept(std::is_nothrow_constructible_v<Iter, U>)
//       : iter_(other.iter_) 
//     {}

//   // Required:
//   [[nodiscard]] constexpr auto dereference() const noexcept(nothrow_dereference<Iter>) -> reference { return *iter_; }
//   constexpr void increment() noexcept(nothrow_increment<Iter>) { 
//     iter_ += (d+1); 
//   }

//   // For forward iterators:
//   template <std::sentinel_for<Iter> S>
//   [[nodiscard]] constexpr auto equals(SimplexLabelIterator<S> const& sentinel) const noexcept(nothrow_equals<Iter, S>)
//     -> bool requires std::forward_iterator<Iter> {
//     return iter_ == sentinel.iter_;
//   }

//   // For bidirectional iterators:
//   constexpr void decrement() noexcept(nothrow_decrement<Iter>) requires std::bidirectional_iterator<Iter> { --iter_; }

//   // For random access iterators:
//   constexpr void advance(difference_type n) noexcept(nothrow_advance<Iter>) requires std::random_access_iterator<Iter> {
//     iter_ += n;
//   }

//   template < std::sized_sentinel_for<Iter> S >
//   [[nodiscard]] constexpr auto distance_to(SimplexLabelIterator<S> const& sentinel) const noexcept(nothrow_distance_to<Iter, S>)
//       -> difference_type requires std::random_access_iterator<Iter> {
//       return sentinel.iter_ - iter_;
//   }
// };

//static_assert(std::forward_iterator<RankLabelIterator>);

// template<typename T, std::size_t N >
// concept ArrayIterator = requires(std::iterator_traits<T> it) {
//   { *it } -> std::same_as<std::array<typename std::iterator_traits<T>::value_type, N>>;
//   { ++it } -> std::same_as<T&>;
//   { it++ } -> std::same_as<T>;
// };

// template <typename T, std::size_t N>
// class SimplexIterator {
// public:
//     using iterator_category = std::input_iterator_tag;
//     using value_type = std::array<T, N>;
//     using difference_type = std::ptrdiff_t;
//     using pointer = value_type*;
//     using reference = value_type&;

//     SimplexIterator(pointer ptr) : ptr_(ptr) {}

//     reference operator*() const { return *ptr_; }
//     pointer operator->() const { return ptr_; }

//     SimplexIterator& operator++() {
//       ++ptr_;
//       return *this;
//     }

//     SimplexIterator operator++(int) {
//       ArrayIterator temp(*this);
//       ++(*this);
//       return temp;
//     }

//     friend bool operator==(const SimplexIterator& lhs, const SimplexIterator& rhs) {
//       return lhs.ptr_ == rhs.ptr_;
//     }

//     friend bool operator!=(const SimplexIterator& lhs, const SimplexIterator& rhs) {
//       return !(lhs == rhs);
//     }

// private:
//     pointer ptr_;
// };




// Given codimension-1 ranks, determines the ranks of the corresponding faces
// p := simplex dimension of given ranks 
auto decompress_faces(const vector< uint_64 >& cr, const size_t n, const size_t p, bool unique) -> vector< uint_64 > {
  vector< uint_64 > fr; 
  fr.reserve(cr.size()); 
  for (auto ci : cr){
    combinatorial::apply_boundary(ci, n, p+1, [&](auto face_rank){ fr.push_back(face_rank); });
  }
  if (unique) {
    std::sort(fr.begin(), fr.end());
    fr.erase(std::unique(fr.begin(), fr.end()), fr.end());
  }
  return fr;
};


template< int p = 0, typename F = double, bool lex_order = true >
struct UpLaplacian {
  static constexpr int dim = p;
  using value_type = F; 
  // typename p dim;
  // using Map_t = IndexMap< uint_64, pthash::murmurhash2_64, pthash::dictionary_dictionary >;
  using Map_t = unordered_map< uint_64, uint_64 >;
  const size_t nv;
  const size_t np;
  const size_t nq; 
  array< size_t, 2 > shape;               // TODO: figure out how to initialize in constructor
  const vector< uint_64 > qr;             // p+1 ranks
  const vector< uint_64 > pr;             // p ranks
  mutable vector< F > y;                  // workspace
  mutable Map_t index_map;                // indexing function
  vector< F > fpl;                        // p-simplex left weights 
  vector< F > fpr;                        // p-simplex right weights 
  vector< F > fq;                         // (p+1)-simplex weights
  vector< F > degrees;                    // weighted degrees; pre-computed

  UpLaplacian(const vector< uint_64 > qr_, const vector< uint_64 > pr_, const size_t nv_) 
    : nv(nv_), np(pr_.size()), nq(qr_.size()), pr(pr_), qr(qr_)  {
    shape = { np, np };
    y = vector< F >(np); // todo: experiment with local _alloca allocation
    fpl = vector< F >(np, 1.0);
    fpr = vector< F >(np, 1.0);
    fq = vector< F >(nq, 1.0); 
    degrees = vector< F >(np, 0.0);
    // pr = decompress_faces(qr, nv, dim+1, true);
    for (uint64_t i = 0; i < pr.size(); ++i){
      index_map.emplace(pr[i], i);
    }
  }

  // Precomputes the degree term
  void precompute_degree(){
    if (fpl.size() != np || fq.size() != nq || fpl.size() != np){ return; }
    std::fill_n(degrees.begin(), degrees.size(), 0);
    size_t q_ind = 0; 
    for (auto qi : qr){
      combinatorial::apply_boundary(qi, nv, dim+2, [&](auto face_rank){ 
        const auto ii = index_map[face_rank];
        degrees.at(ii) += fpl.at(ii) * fq.at(q_ind) * fpr.at(ii);
      });
      q_ind += 1; 
    }
  };

  // Takes as input a range [b, e) representing the face labels, return their index 
  template< typename It > 
  size_t index(It b, const It e) const {
    // auto face_rank = std::array< uint16_t, 1 >();
    uint16_t face_rank = 0; 
    lex_rank(b, e, nv, dim+1, &face_rank);
    return index_map[face_rank];
  };

  // Takes as input a range [b,e) of labels of a q-simplex, returns a (d+1) tuple of 
  // of indices (i,j,k,...) representing the indexes of the faces in its boundary
  template< std::input_iterator It, std::sentinel_for< It > Sen, typename OutputIt >
  void boundary_indices(It b, Sen e, OutputIt out) const {     
    const size_t d = std::distance(b, e)-1;   
    for_each_combination(b, b+d, e, [&](auto pb, auto pe){
      *out = this->index(pb, pe);
      ++out;
      return false; 
    });
  }

  // Internal matvec; outputs y = L @ x
  inline void __matvec(F* x) const noexcept { 
    // Start with the degree computation
    std::transform(degrees.begin(), degrees.end(), x, y.begin(), std::multiplies< F >());

    // The matvec
    size_t q_ind = 0;
    auto p_ranks = array< uint64_t, dim+2 >();
    auto p_ind = array< uint32_t, dim+2 >();

    // #pragma omp simd
    for (auto qi: qr){
      if constexpr (p == 0){
				lex_unrank_2(static_cast< I >(qi), static_cast< I >(nv), begin(p_ranks));
        const auto ii = p_ranks[0]; // index_map[q_vertices[0]]; 
        const auto jj = p_ranks[1];
        y[ii] -= x[jj] * fpl[ii] * fq[q_ind] * fpr[jj]; 
        y[jj] -= x[ii] * fpl[jj] * fq[q_ind] * fpr[ii]; 
      } else if constexpr (p == 1) { // inject the sign @ compile time
        // boundary(qi, nv, p+2, begin(p_ranks));
        
        auto q = lex_unrank(qi, nv, p+2);
        boundary_indices(q.begin(), q.end(), p_ind.begin());

        const auto ii = p_ind[0]; // index_map[p_ranks[0]];
        const auto jj = p_ind[1]; // index_map[p_ranks[1]];
        const auto kk = p_ind[2]; // index_map[p_ranks[2]];
        y[ii] -= x[jj] * fpl[ii] * fq[q_ind] * fpr[jj];
        y[jj] -= x[ii] * fpl[jj] * fq[q_ind] * fpr[ii]; 
        y[ii] += x[kk] * fpl[ii] * fq[q_ind] * fpr[kk]; 
        y[kk] += x[ii] * fpl[kk] * fq[q_ind] * fpr[ii]; 
        y[jj] -= x[kk] * fpl[jj] * fq[q_ind] * fpr[kk]; 
        y[kk] -= x[jj] * fpl[kk] * fq[q_ind] * fpr[jj]; 
      } else {
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
}; // UpLaplacian

