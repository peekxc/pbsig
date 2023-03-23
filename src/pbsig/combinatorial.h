// combinatorial.h 
// Contains routines for combinatorics-related tasks 
// The combinations and permutations generation code is copyright Howard Hinnant, taken from: https://github.com/HowardHinnant/combinations/blob/master/combinations.h
#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H 

#include <cstdint>		// uint_fast64_t
#include <array>
// #include <span> 		 	// span (C++20)
#include <cmath>	 	 	// round, sqrt
#include <numeric>   	// midpoint, accumulate
#include <vector> 	 	// vector  
#include <algorithm> 
#include <type_traits>
#include <vector>
#include <functional>
#include <iterator>

using std::begin;
using std::end; 
using std::vector; 
using std::size_t;

namespace combinatorial {
	using I = uint_fast64_t;

	template<typename T>
	using it_diff_t = typename std::iterator_traits<T>::difference_type;

	// Rotates two discontinuous ranges to put *first2 where *first1 is.
	//     If last1 == first2 this would be equivalent to rotate(first1, first2, last2),
	//     but instead the rotate "jumps" over the discontinuity [last1, first2) -
	//     which need not be a valid range.
	//     In order to make it faster, the length of [first1, last1) is passed in as d1,
	//     and d2 must be the length of [first2, last2).
	//  In a perfect world the d1 > d2 case would have used swap_ranges and
	//     reverse_iterator, but reverse_iterator is too inefficient.
	template <class It>
	void rotate_discontinuous(
		It first1, It last1, it_diff_t< It > d1,
		It first2, It last2, it_diff_t< It > d2)
	{
		using std::swap;
		if (d1 <= d2){ std::rotate(first2, std::swap_ranges(first1, last1, first2), last2); }
		else {
			It i1 = last1;
			while (first2 != last2)
				swap(*--i1, *--last2);
			std::rotate(first1, i1, last1);
		}
	}

	// Call f() for each combination of the elements [first1, last1) + [first2, last2)
	//    swapped/rotated into the range [first1, last1).  As long as f() returns
	//    false, continue for every combination and then return [first1, last1) and
	//    [first2, last2) to their original state.  If f() returns true, return
	//    immediately.
	//  Does the absolute mininum amount of swapping to accomplish its task.
	//  If f() always returns false it will be called (d1+d2)!/(d1!*d2!) times.
	template < typename It, typename Lambda >
	bool combine_discontinuous(
		It first1, It last1, it_diff_t< It > d1,  
		It first2, It last2, it_diff_t< It > d2,
		Lambda&& f, it_diff_t< It > d = 0)
	{
		using D = it_diff_t< It >;
		using std::swap;
		if (d1 == 0 || d2 == 0){ return f(); }
		if (d1 == 1) {
			for (It i2 = first2; i2 != last2; ++i2) {
				if (f()){ return true; }
				swap(*first1, *i2);
			}
		}
		else {
			It f1p = std::next(first1), i2 = first2;
			for (D d22 = d2; i2 != last2; ++i2, --d22){
				if (combine_discontinuous(f1p, last1, d1-1, i2, last2, d22, f, d+1))
					return true;
				swap(*first1, *i2);
			}
		}
		if (f()){ return true; }
		if (d != 0){ rotate_discontinuous(first1, last1, d1, std::next(first2), last2, d2-1); }
		else { rotate_discontinuous(first1, last1, d1, first2, last2, d2); }
		return false;
	}

	template < typename Lambda, typename It > 
	struct bound_range { 
		Lambda f_;
		It first_, last_;
		bound_range(Lambda& f, It first, It last) : f_(f), first_(first), last_(last) {}
		bool operator()(){ return f_(first_, last_); } 
		bool operator()(It, It) { return f_(first_, last_); }
	};

	template <class It, class Function>
	Function for_each_combination(It first, It mid, It last, Function&& f) {
		bound_range<Function&, It> wfunc(f, first, mid);
		combine_discontinuous(first, mid, std::distance(first, mid),
													mid, last, std::distance(mid, last),
													wfunc);
		return std::move(f);
	}

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
	
	// Baseline from: https://stackoverflow.com/questions/44718971/calculate-binomial-coffeficient-very-reliably
	// Requires O(min{k,n-k}), uses pascals triangle approach (+ degenerate cases)
	constexpr inline size_t binom(size_t n, size_t k) noexcept {
		return
			(        k> n  )? 0 :          // out of range
			(k==0 || k==n  )? 1 :          // edge
			(k==1 || k==n-1)? n :          // first
			(     k+k < n  )?              // recursive:
			(binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
			(binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
	}

	// Non-cached version of the binomial coefficient using floating point algorithm
	// Requires O(k), uses very simple loop
	[[nodiscard]]
	inline size_t binomial_coeff_(const double n, const size_t k) noexcept {
	  double bc = n;
	  for (size_t i = 2; i <= k; ++i){ bc *= (n+1-i)/i; }
	  return(static_cast< size_t >(std::round(bc)));
	}

	// Table to cache low values of the binomial coefficient
	template< size_t max_n, size_t max_k, typename value_t = size_t >
	struct BinomialCoefficientTable {
	  value_t combinations[max_k][max_n+1];
	  constexpr BinomialCoefficientTable() : combinations() {
			auto n_dispatcher = make_index_dispatcher< max_n+1 >();
			auto k_dispatcher = make_index_dispatcher< max_k >();
			n_dispatcher([&](auto i) {
				k_dispatcher([&](auto j){
					combinations[j][i] = bc_recursive< i, j >();
				});
			});
	  }

		// Evaluate general binomial coefficient, using cached table if possible 
		value_t operator()(const value_t n, const value_t k) const {
			if (k == 0 || n == k){ return 1; }
			if (n < k){ return 0; }
			if (k == 2){ return static_cast< value_t >((n*(n-1))/2); }
			if (k == 1){ return n; }
			// return binom(n, k);
			// return binomial_coeff_(n,std::min(k,n-k));
			return (n < max_n && k < max_k) ? combinations[k][n] : binomial_coeff_(n,std::min(k,n-k));
		}
	};
	// Build the cached table
	static constexpr auto BC = BinomialCoefficientTable< 64, 3 >();
	
	// Wrapper to choose between cached and non-cached version of the Binomial Coefficient
	constexpr size_t BinomialCoefficient(const size_t n, const size_t k){
	  return BC(n,k);
	}
	
	#if __cplusplus >= 202002L
    // C++20 (and later) code
		// constexpr midpoint midpoint
		using std::midpoint; 
	#else
		template < class Integer > 
		constexpr Integer midpoint(Integer a, Integer b) noexcept {
			return (a+b)/2;
		}
	#endif

	// All inclusive range binary search 
	// Compare must return -1 for <(key, index), 0 for ==(key, index), and 1 for >(key, index)
	// Guaranteed to return an index in [0, n-1] representing the lower_bound
	template< typename T, typename Compare > [[nodiscard]]
	int binary_search(const T key, size_t n, Compare p) noexcept {
	  int low = 0, high = n - 1, best = 0; 
		while( low <= high ){
			int mid = int{ midpoint(low, high) };
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
	
	[[nodiscard]]
	constexpr auto rank_colex_2(I i, I j) noexcept {
		return BinomialCoefficient(i, 2) + BinomialCoefficient(j, 1);
	}

	// Lexicographically rank 2-subsets
	[[nodiscard]]
	constexpr auto rank_lex_2(I i, I j, const I n) noexcept {
	  if (j < i){ std::swap(i,j); }
	  return I(n*i - i*(i+1)/2 + j - i - 1);
	}

	#include <iostream> 
	// Lexicographically rank k-subsets
	template< typename InputIter >
	[[nodiscard]]
	inline I rank_lex_k(InputIter s, const size_t n, const size_t k, const I N){
		I i = k; 
		// std::cout << std::endl; 
	  const I index = std::accumulate(s, s+k, 0, [n, &i](I val, I num){ 
			// std::cout << BinomialCoefficient((n-1) - num, i) << ", ";
		  return val + BinomialCoefficient((n-1) - num, i--); 
		});
		std::cout << std::endl; 
	  const I combinadic = (N-1) - index; // Apply the dual index mapping
	  return combinadic;
	}

	// Colexicographically rank k-subsets
	// assumes each k tuple of s is in colex order! 
	template< bool colex = true, typename InputIter >
	[[nodiscard]]
	constexpr auto rank_colex_k(InputIter s, const size_t k) noexcept {
		if constexpr(colex){
			I i = k; 
			const I index = std::accumulate(s, s+k, 0, [&i](I val, I num){ 
				return val + BinomialCoefficient(num, i--); 
			});
			return index; 
		} else {
			I i = 1; 
			const I index = std::accumulate(s, s+k, 0, [&i](I val, I num){ 
				return val + BinomialCoefficient(num, i++); 
			});
			return index; 
		}
	}

	// Rank a stream of integers (lexicographically)
	template< typename InputIt, typename OutputIt >
	inline void rank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		switch (k){
			case 2:{
				for (; s != e; s += k){
					*out++ = rank_lex_2(*s, *(s+1), n);
				}
				break;
			}
			default: {
				const I N = BinomialCoefficient(n, k); 
				for (; s != e; s += k){
					*out++ = rank_lex_k(s, n, k, N);
				}
				break;
			}
		}
	}

	// Rank a stream of integers (colexicographically)
	template< bool colex = true, typename InputIt, typename OutputIt >
	inline void rank_colex(InputIt s, const InputIt e, const size_t k, OutputIt out){
		for (; s != e; s += k){
			*out++ = rank_colex_k< colex >(s, k);
		}
	}

	// Lexicographically unrank 2-subsets
	template< typename OutputIt  >
	inline auto unrank_lex_2(const I r, const I n, OutputIt out) noexcept  {
		auto i = static_cast< I >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
		auto j = static_cast< I >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
		*out++ = i; // equivalent to *out = i; ++i;
		*out++ = j; // equivalent to *out = j; ++j;
	}
	
	// Lexicographically unrank k-subsets [ O(log n) version ]
	template< typename OutputIterator > 
	inline void unrank_lex_k(I r, const size_t n, const size_t k, OutputIterator out) noexcept {
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

	// Lexicographically unrank subsets wrapper
	template< bool lex = true, typename InputIt, typename OutputIt >
	inline void unrank_lex(InputIt s, const InputIt e, const size_t n, const size_t k, OutputIt out){
		switch(k){
			case 2: 
				for (; s != e; ++s){
					unrank_lex_2(*s, n, out);
				}
				break;
			default:
				for (; s != e; ++s){
					unrank_lex_k(*s, n, k, out);
				}
				break;
		}
	}


	template <class Predicate>
	I get_max(I top, const I bottom, const Predicate pred) {
		if (!pred(top)) {
			I count = top - bottom;
			while (count > 0) {
				I step = count >> 1, mid = top - step;
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

	I get_max_vertex(const I idx, const I k, const I n) {
		return get_max(n, k - 1, [&](I w) -> bool { return BinomialCoefficient(w, k) <= idx; });
	}

	template < typename InputIt, typename OutputIt >
	void unrank_colex(InputIt s, const InputIt e, const I k, const I n, OutputIt out) {
		for (I N = n - 1; s != e; ++s, N = n - 1){
			I r = *s; 
			for (I d = k; d > 1; --d) {
				N = get_max_vertex(r, d, N);
				std::cout << "r: " << r << ", d: " << d << ", N: " << N << std::endl;
				*out++ = N;
				r -= BinomialCoefficient(N, d);
			}
			*out++ = r;
		}
	}

	// Lexicographically unrank subsets wrapper
	// template< bool colex = true, typename InputIt, typename OutputIt >
	// inline void unrank_colex(InputIt s, const InputIt e, const size_t k, OutputIt out){
	// 	switch(k){
	// 		case 2: 
	// 			for (; s != e; ++s){
	// 				unrank_lex_2(*s, n, out);
	// 			}
	// 			break;
	// 		default:
	// 			for (; s != e; ++s){
	// 				unrank_lex_k(*s, n, k, out);
	// 			}
	// 			break;
	// 	}
	// }

} // namespace combinatorial




	


//   // Lexicographically unrank subsets wrapper
// 	template< size_t k, typename InputIt, typename Lambda >
// 	inline void lex_unrank_f(InputIt s, const InputIt e, const size_t n, Lambda f){
//     if constexpr (k == 2){
//       std::array< I, 2 > edge;
//       for (; s != e; ++s){
//         lex_unrank_2(*s, n, edge.begin());
//         f(edge);
//       } 
//     } else if (k == 3){
//       std::array< I, 3 > triangle;
//       for (; s != e; ++s){
//         lex_unrank_k(*s, n, 3, triangle.begin());
// 				f(triangle);
//       }
// 		} else {
//       std::array< I, k > simplex;
//       for (; s != e; ++s){
//         lex_unrank_k(*s, n, k, simplex.begin());
// 				f(simplex);
//       }
//     }
// 	}

	
// 	[[nodiscard]]
// 	inline auto lex_unrank_2_array(const I r, const I n) noexcept -> std::array< I, 2 > {
// 		auto i = static_cast< I >( (n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)) );
// 		auto j = static_cast< I >( r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 );
// 		return(std::array< I, 2 >{ i, j });
// 	}
	
// 	[[nodiscard]]
// 	inline auto lex_unrank(const size_t rank, const size_t n, const size_t k) -> std::vector< I > {
// 		if (k == 2){
// 			auto a = lex_unrank_2_array(rank, n);
// 			std::vector< I > out(a.begin(), a.end());
// 			return(out);
// 		} else {
// 			std::vector< I > out; 
// 			out.reserve(k);
// 			lex_unrank_k(rank, n, k, std::back_inserter(out));
// 			return(out);
// 		}
// 	}
	

// 	template< typename Lambda >
// 	void apply_boundary(const size_t r, const size_t n, const size_t k, Lambda f){
// 		// Given a p-simplex's rank representing a tuple of size p+1, enumerates the ranks of its (p-1)-faces, calling Lambda(*) on its rank
// 		using combinatorial::I; 
// 		switch(k){
// 			case 0: { return; }
// 			case 1: {
// 				f(r);
// 				return;
// 			}
// 			case 2: {
// 				auto p_vertices = std::array< I, 2 >();
// 				lex_unrank_2(static_cast< I >(r), static_cast< I >(n), begin(p_vertices));
// 				f(p_vertices[0]);
// 				f(p_vertices[1]);
// 				return;
// 			}
// 			case 3: {
// 				auto p_vertices = std::array< I, 3 >();
// 				lex_unrank_k(r, n, k, begin(p_vertices));
// 				f(lex_rank_2(p_vertices[0], p_vertices[1], n));
// 				f(lex_rank_2(p_vertices[0], p_vertices[2], n));
// 				f(lex_rank_2(p_vertices[1], p_vertices[2], n));
// 				return; 
// 			} 
// 			default: {
// 				auto p_vertices = std::vector< I >(0, k);
// 				lex_unrank_k(r, n, k, p_vertices.begin());
// 				const I N = BinomialCoefficient(n, k); 
// 				combinatorial::for_each_combination(begin(p_vertices), begin(p_vertices)+2, end(p_vertices), [&](auto a, auto b){
// 					f(lex_rank_k(a, n, k, N));
// 					return false; 
// 				});
// 				return; 
// 			}
// 		}
// 	} // apply boundary 

// 	template< typename OutputIt >
// 	void boundary(const size_t p_rank, const size_t n, const size_t k, OutputIt out){
// 		apply_boundary(p_rank, n, k, [&out](auto face_rank){
// 			*out = face_rank;
// 			out++;
// 		});
// 	}
// } // namespace combinatorial

	// // Lexicographically unrank k-subsets
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

#endif 