#include <concepts>
#include <span>
#include <cstddef>
#include <optional>

using std::optional;
using std::size_t;
using std::pair;

template< typename M, typename F = typename M::value_type >
concept Pivotable = requires(M a) {
  { a.low(size_t(0)) } -> std::same_as< optional< pair< size_t, F > > >;
	{ a.low_index(size_t(0)) } -> std::same_as< optional< size_t > >;
	{ a.low_value(size_t(0)) } -> std::same_as< optional< F > >;
};

template< typename M, typename F = typename M::value_type >
concept Addable = requires(M a) {
	{ a.scale_col(size_t(0), F(0)) } -> std::same_as< void >;
	{ a.add_cols(size_t(0), size_t(0), size_t(0)) } -> std::same_as< void >;
	{ a.add_scaled_col(size_t(0), size_t(0), size_t(0), F(0)) } -> std::same_as< void >;
};

// A type M is said to cancellable if it supports a cancel_lowest() method
// where cancel_lowest(i,j) => lowest entry of column i is cancelled using lowest entry of column j
// Only the method name is required here, since checking the semantics would have to be done at run-time
template< typename M, typename F = typename M::value_type >
concept Cancellable = requires(M a) {
	{ a.cancel_lowest(size_t(0), size_t(0)) } -> std::same_as< void >;
};

// A ReducibleMatrix:
//	- has a dim() method that returns a pair of size-types  
//  - has a find_low() method that accepts two indices and returns an optional pair 
//  - satisfies Pivotable and Addable
template< typename M, typename F = typename M::value_type >
concept ReducibleMatrix = requires(M a){
	{ a.dim() } -> std::same_as< pair< size_t, size_t > >;
	{ a.find_low(size_t(0), size_t(0)) } -> std::same_as< std::optional< pair< size_t, F > > >; 
} && Pivotable< M, F > && Addable< M, F > && Cancellable< M, F >;


// A PermutableMatrix:
//  - has both a swap_rows(i,j) and a swap_cols(i,j) method
//  - has a permute_rows(...) and permute_cols(...) methods where the input ... is convertible to a span< size_t >
//  - has a column_empty() method
//  - satisfies ReducibleMatrix
template< typename M, typename F = typename M::value_type >
concept PermutableMatrix = requires(M a){
	{ a.swap_rows(size_t(0), size_t(0)) } -> std::same_as< void >;
	{ a.swap_cols(size_t(0), size_t(0)) } -> std::same_as< void >;
	{ a.swap(size_t(0), size_t(0)) } -> std::same_as< void >;
	{ a.permute_rows(std::span< size_t >()) } -> std::same_as< void >;
	{ a.permute_cols(std::span< size_t >()) } -> std::same_as< void >;
	{ a.column_empty(size_t(0)) } -> std::same_as< bool >;
	{ a(size_t(0), size_t(0)) } -> std::same_as< F >; 
} && ReducibleMatrix< M, F >;
