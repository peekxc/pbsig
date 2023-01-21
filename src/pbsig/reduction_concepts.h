#include <concepts>
#include <span>
#include <cstddef>
#include <optional>

using std::optional;
using std::size_t;
using std::pair;

 // iadd_cols(i,j) <=> col(i) <- col(i) + col(j) 
 // iadd_scaled_col(i,j,s) <=> col(i) <- col(i) + s*col(j)
template< typename M, typename F = typename M::value_type >
concept Addable = requires(M a) {
	{ a.scale_col(size_t(0), F(0)) } -> std::same_as< void >;
	{ a.iadd_cols(size_t(0), size_t(0)) } -> std::same_as< void >;
	{ a.iadd_scaled_col(size_t(0), size_t(0), F(0)) } -> std::same_as< void >;
};

// A type M is said to cancellable if it supports a cancel_lowest() method
// where cancel_lowest(i,j) => lowest entry of column i is cancelled using lowest entry of column j
// template< typename M, typename F = typename M::value_type >
// concept Cancellable = requires(M a) {
// 	{ a.cancel_lowest(size_t(0), size_t(0)) } -> std::same_as< void >;
// };

// Need notion of a 'column' 

// A ReducibleMatrix:
//	- has a dim() method that returns a pair of size-types  
// 	- has efficient access to its lowest non-zero entries (both their row indices and their values)
//  - satisfies Pivotable and Addable
template< typename M, typename F = typename M::value_type >
concept ReducibleMatrix = requires(M a){
	{ a.dim() } -> std::same_as< pair< size_t, size_t > >;
	{ a.low(size_t(0)) } -> std::same_as< optional< pair< size_t, F > > >;
	{ a.low_index(size_t(0)) } -> std::same_as< optional< size_t > >;
	{ a.low_value(size_t(0)) } -> std::same_as< optional< F > >;
	{ a.clear_column(size_t(0)) } -> std::same_as< void >; // for the clearing optimization		
	// { a.find_low(size_t(0), size_t(0)) } -> std::same_as< std::optional< pair< size_t, F > > >; 
} && Addable< M, F >;


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
	{ a.operator()(size_t(0), size_t(0)) } -> std::same_as< F >; 
	{ a.cancel_lowest(size_t(0), size_t(0)) } -> std::same_as< void >; // to simplify the code greatly
} && ReducibleMatrix< M, F >;
