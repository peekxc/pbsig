#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H 

#include <cstdint>
#include <array>
#include <span>
#include <cmath>	 // round, sqrt
#include <numeric>   // midpoint, accumulate

namespace combinatorial {
	using I = uint_fast64_t;

	template <std::size_t... Idx>
	constexpr auto make_index_dispatcher(std::index_sequence<Idx...>) {
		return [] (auto&& f) { (f(std::integral_constant<std::size_t,Idx>{}), ...); };
	};
	
	template <std::size_t N>
	constexpr auto make_index_dispatcher() {
		return make_index_dispatcher(std::make_index_sequence< N >{});
	};
	
	// Constexpr binomial coefficient using recursive formulation
	template < size_t n, size_t k >
	constexpr auto bc_recursive() noexcept {
		if constexpr ( n == k || k == 0 ){ return(1); }
		else if constexpr (n == 0 || k > n){ return(0); }
		else {
		 return (n * bc_recursive< n - 1, k - 1>()) / k;
		}
	}
	
	// Table to cache low values of the binomial coefficient
	template< size_t n, size_t k >
	struct BinomialCoefficientTable {
	  size_t combinations[n+1][k];
	  constexpr BinomialCoefficientTable() : combinations() {
			auto n_dispatcher = make_index_dispatcher< n+1 >();
			auto k_dispatcher = make_index_dispatcher< k >();
			n_dispatcher([&](auto i) {
				k_dispatcher([&](auto j){
					combinations[i][j] = bc_recursive< i, j >();
				});
			});
	  }
	};
	
	// Build the cached table
	static constexpr size_t max_choose = 16;
	static constexpr auto BC = BinomialCoefficientTable< max_choose, max_choose >();
	
	// Non-cached version of the binomial coefficient using floating point algorithm
	[[nodiscard]]
	inline size_t binomial_coeff_(const double n, const size_t k) noexcept {
	  double bc = n;
	  for (size_t i = 2; i <= k; ++i){ bc *= (n+1-i)/i; }
	  return(static_cast< size_t >(std::round(bc)));
	}
	
	// Wrapper to choose between cached and non-cached version of the Binomial Coefficient
	inline size_t BinomialCoefficient(const size_t n, const size_t k){
	  if (k == 0 || n == k){ return 1; }
	  if (n < k){ return 0; }
	  if (k == 2){ return((n*(n-1))/2); }
	  return n < max_choose ? BC.combinations[n][k] : binomial_coeff_(n,std::min(k,n-k));
	}
	
	// All inclusive range binary search 
	// Compare must return -1 for <(key, index), 0 for ==(key, index), and 1 for >(key, index)
	// Guaranteed to return an index in [0, n-1] representing the lower_bound
	template< typename T, typename Compare > [[nodiscard]]
	int binary_search(const T key, size_t n, Compare p) {
	  int low = 0, high = n - 1, best = 0; 
		while( low <= high ){
			int mid = int{ std::midpoint(low, high) };
			auto cmp = p(key, mid);
			if (cmp == 0){ 
				while(p(key, mid + 1) == 0){ ++mid; }
				return(mid);
			}
			if (cmp < 0){ high = mid - 1; } 
			else { 
				low = mid + 1; 
				best = mid; 
			}
		}
		return(best);
	}
	
	// ----- Combinatorial Number System functions -----

	// Lexicographically rank 2-subsets
	[[nodiscard]]
	constexpr auto lex_rank_2(I i, I j, const I n) noexcept {
	  if (j < i){ std::swap(i,j); }
	  return I(n*i - i*(i+1)/2 + j - i - 1);
	}
	
	// Lexicographically unrank 2-subsets
	template< typename OutputIt  >
	inline auto lex_unrank_2(const I r, const I n, OutputIt out) noexcept  {
		auto i = static_cast< I >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
		auto j = static_cast< I >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
		*out++ = i; // equivalent to *out = i; ++i;
		*out++ = j; // equivalent to *out = j; ++j;
	}
	
	// Lexicographically unrank k-subsets
	// template< typename OutputIterator >
	// inline void lex_unrank_k(I r, const size_t k, const size_t n, OutputIterator out){
	// 	auto subset = std::vector< size_t >(k); 
	// 	size_t x = 1; 
	// 	for (size_t i = 1; i <= k; ++i){
	// 		while(r >= BinomialCoefficient(n-x, k-i)){
	// 			r -= BinomialCoefficient(n-x, k-i);
	// 			x += 1;
	// 		}
	// 		*out++ = (x - 1);
	// 		x += 1;
	// 	}
	// }
	
	// Lexicographically unrank k-subsets [ O(log n) version ]
	template< typename OutputIterator > 
	inline void lex_unrank_k(I r, const size_t k, const size_t n, OutputIterator out) noexcept {
		const size_t N = combinatorial::BinomialCoefficient(n, k);
		r = (N-1) - r; 
		// auto S = std::vector< size_t >(k);
		for (size_t ki = k; ki > 0; --ki){
			int offset = binary_search(r, n, [ki](const auto& key, int index) -> int {
				auto c = combinatorial::BinomialCoefficient(index, ki);
				return(key == c ? 0 : (key < c ? -1 : 1));
			});
			r -= combinatorial::BinomialCoefficient(offset, ki); 
			*out++ = (n-1) - offset;
		}
	}

	// Lexicographically rank k-subsets
	template< typename InputIter >
	[[nodiscard]]
	inline I lex_rank_k(InputIter s, const size_t k, const size_t n, const I N){
		I i = k; 
	  const I index = std::accumulate(s, s+k, 0, [n, &i](I val, I num){ 
		  return val + BinomialCoefficient((n-1) - num, i--); 
		});
	  const I combinadic = (N-1) - index; // Apply the dual index mapping
	  return(combinadic);
	}
	
	// Lexicographically unrank subsets wrapper
	template< typename InputIt, typename OutputIt >
	inline void lex_unrank(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		for (; s != e; ++s){
			switch(k){
				case 2: 
					lex_unrank_2(*s, n, out);
					break;
				default:
					lex_unrank_k(*s, k, n, out);
					break;
			}
		}
	}

  // Lexicographically unrank subsets wrapper
	template< size_t k, typename InputIt, typename Lambda >
	inline void lex_unrank_f(InputIt s, const InputIt e, const size_t n, Lambda f){
    if constexpr (k == 2){
      std::array< I, 2 > edge;
      for (; s != e; ++s){
        lex_unrank_2(*s, n, edge.begin());
        f(edge);
      } 
    } else if (k == 3){
      std::array< I, 3 > triangle;
      for (; s != e; ++s){
        lex_unrank_k(*s, 3, n, triangle.begin());
      }
		} else {
      std::array< I, k > simplex;
      for (; s != e; ++s){
        lex_unrank_k(*s, k, n, simplex.begin());
      }
    }
	}

	
	// Lexicographically rank subsets wrapper
	template< typename InputIt, typename OutputIt >
	inline void lex_rank(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		if (k == 2){
			for (; s != e; s += k){
				*out++ = lex_rank_2(*s, *(s+1), n);
			}
		} else {
			const I N = BinomialCoefficient(n, k); 
			for (; s != e; s += k){
				*out++ = lex_rank_k(s, k, n, N);
			}
		}
	}
	
	template< typename T > [[nodiscard]]
	inline T lex_rank(std::span< T > s, const size_t n){
		static_assert(std::is_integral_v< T >, "T must be integral type.");
		switch(s.size()){
			case 2: 
				return(lex_rank_2(s[0], s[1], n));
				break;
			default: 
				return(lex_rank_k(s.begin(), s.size(), n, BinomialCoefficient(n, s.size())));
				break;
		}
	}
	
	[[nodiscard]]
	inline auto lex_unrank_2_array(const I r, const I n) noexcept -> std::array< I, 2 > {
		auto i = static_cast< I >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
		auto j = static_cast< I >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
		return(std::array< I, 2 >{ i, j });
	}
	
	[[nodiscard]]
	inline auto lex_unrank(const size_t rank, const size_t n, const size_t k) -> std::vector< I > {
		if (k == 2){
			auto a = lex_unrank_2_array(rank, n);
			std::vector< I > out(a.begin(), a.end());
			return(out);
		} else {
			std::vector< I > out; 
			out.reserve(k);
			lex_unrank_k(rank, k, n, std::back_inserter(out));
			return(out);
		}
	}
	
	
	
} // namespace combinatorial

template< class Iter, class Index >
auto cycle_leader_shortcut(Iter t, Index start){
  auto cur = t[start];
  while(cur != start){
    if (cur < start) return false;
    cur = t[cur];
  }
  return true;
}
template< class Iter, class Index >
void reverse_cycle(Iter t, Index start){
  auto cur = t[start];
  auto prev = start;
  while( cur != start ){
    auto next = t[cur];
    t[cur] = prev;
    prev = cur;
    cur = next;
  }
  t[start] = prev;
}

// O(n log n) in-place permutation inverse algorithm
// Based on: https://stackoverflow.com/questions/56603153/how-to-invert-a-permutation-represented-by-an-array-in-place
template< class Iter >
inline void inverse_permutation_n(Iter first, size_t n){
  for (size_t i = 0; i != n; ++i){
		if (cycle_leader_shortcut(first, i)){ reverse_cycle(first, i); };
  }
}


// Schwartzian transform
// From: https://devblogs.microsoft.com/oldnewthing/20170106-00/?p=95135
template<typename Iter, typename UnaryOperation, typename Compare>
void sort_by_with_caching(Iter first, Iter last, UnaryOperation op, Compare comp) {
	using Diff = typename std::iterator_traits<Iter>::difference_type;
	using T = typename std::iterator_traits<Iter>::value_type;
	using Key = decltype(op(std::declval<T>()));
	using Pair = std::pair<T, Key>;
	Diff length = std::distance(first, last);
	
	// Construct the "cache" by applying op to every T 
	std::vector< Pair > pairs;
	pairs.reserve(length);
	std::transform(first, last, std::back_inserter(pairs), [&](T& t) { 
		return std::make_pair(std::move(t), op(t)); 
	});
	
	// Now do the cached comparison-based sort  
	std::sort(pairs.begin(), pairs.end(), [&](const Pair& a, const Pair& b) { 
		return comp(a.second, b.second); 
	});
	// ... and move the resulting sorted keys back into the range
	std::transform(pairs.begin(), pairs.end(), first, [](Pair& p) { return std::move(p.first); });
}

// Applies a permutation (indices) in O(n) time in-place - Essentially performs cycle sort
// Based on: https://devblogs.microsoft.com/oldnewthing/20170104-00/?p=95115
template< typename Iter1, typename Iter2 >
void apply_permutation(Iter1 b, Iter1 e, Iter2 p) {
	using T = typename std::iterator_traits< Iter1 >::value_type;
	using Diff = typename std::iterator_traits< Iter2 >::value_type;
	const Diff length = std::distance(b, e);
	for (Diff i = 0; i < length; ++i) {
		Diff current = i;
		if (i != p[current]) {				// Don't process trivial cycles
			T element{std::move(b[i])};
			while (i != p[current]) {   // Sort the current non-trivial cycle
				Diff next = p[current];
				b[current] = std::move(b[next]);
				p[current] = current;
				current = next;
			}
			b[current] = std::move(element);
			p[current] = current;
		}
	}
}

// Applies the reverse of the permutation given by 'indices' to the range [first, last)
template< typename Iter1, typename Iter2 >
void apply_reverse_permutation(Iter1 first, Iter1 last, Iter2 indices) {
	using T = typename std::iterator_traits<Iter1>::value_type;
	using Diff = typename std::iterator_traits<Iter2>::value_type;
	const Diff length = std::distance(first, last);
	for (Diff i = 0; i < length; ++i) {
		while (i != indices[i]) {
			Diff next = indices[i];
			if (next < 0 || next >= length) { throw std::range_error("Invalid index in permutation"); }
	    if (next == indices[next]) { throw std::range_error("Not a permutation"); }
			std::swap(first[i], first[next]);
			std::swap(indices[i], indices[next]);
		}
	}
}

// Creates and returns the inverse permutation of p 
template< typename Iter >
std::vector< size_t > inverse_permutation(Iter b, const Iter e) {
	auto a = std::vector< size_t >(b, e);
	inverse_permutation_n(a.begin(), a.size());
	return(a);
}

#endif 