#include <string>
#include <cstddef>
#include <concepts>
#include <iostream>
#include <optional>
#include <tuple>
#include <algorithm> 
#include <numeric> // iota
#include <vector> 
#include "reduction_concepts.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <pybind11/eigen.h>
#include <Eigen/Sparse>

using std::pair; 
using std::make_pair;
using std::size_t; 
using std::optional;
using std::tuple;
using std::make_tuple;
using std::vector; 
// using VectorXf = Eigen::VectorXf;

// Global static variable to store statistics
// [0] stores number of column operations 
// [1] stores number of field operations, if possible
static std::array< size_t, 2 > reduction_stats; 

template< typename F > 
constexpr bool equals_zero(F val) noexcept {
  if constexpr (std::is_integral_v< F >){
    return val == 0; 
  } else {
    return std::abs(val) <= std::numeric_limits< F >::epsilon();
  }
}

[[nodiscard]]
inline auto move_right_permutation(size_t i, size_t j, const size_t n) -> std::vector< size_t > {
  if (i > j){ throw std::invalid_argument("invalid");}
  std::vector< size_t > v(n);
  std::iota(v.begin(), v.end(), 0);
  std::rotate(v.begin()+i, v.begin()+i+1, v.begin()+j+1);
  return(v);
}

[[nodiscard]]
inline auto move_left_permutation(size_t i, size_t j, const size_t n) -> std::vector< size_t >{
  if (i < j){ throw std::invalid_argument("invalid");}
  std::vector< size_t > v(n);
  std::iota(v.begin(), v.end(), 0);
  std::rotate(v.rbegin()+(n-(i+1)), v.rbegin()+(n-i), v.rbegin()+(n-j));
  return(v);
}


template< bool clearing = true, ReducibleMatrix Matrix, typename Iter, typename Lambda >
void pHcol_local(Matrix& R1, Matrix& V1, Matrix& R2, Matrix& V2, Iter b1, const Iter e1, Iter b2, const Iter e2, Lambda f){
	using field_t = typename Matrix::value_type;
	using entry_t = typename std::pair< size_t, field_t >;

	// Reduce (R2,V2)
	pHcol(R2, V2, b2, e2, f);

	// Apply clearing optimization
	if constexpr(clearing){
		optional< entry_t > low_j;
		for (size_t j; b2 != e2; ++b2){
			j = *b2; 
			if (!R2.column_empty(j) && (low_j = R2.low(j))){
				R1.clear_column(low_j->first);
				auto R_j = std::vector< entry_t >();
				R2.column(j, [&](auto row_idx, auto v){ R_j.push_back(std::make_pair(row_idx, v)); });
				V1.assign_column(low_j->first, R_j.begin(), R_j.end());
			}
		}
	}
	
	// Now reduce (R1, V1)
	pHcol(R1, V1, b1, e1, f);
	return;
}


// https://www.albany.edu/~ML644186/AMAT_840_Spring_2019/Math840_Notes.pdf
template< ReducibleMatrix Matrix, typename Iter, typename Lambda >
void pHcol(Matrix& R, Matrix& V, Iter b, const Iter e, Lambda f){
	using field_t = typename Matrix::value_type;
	using entry_t = typename std::pair< size_t, field_t >;
	
	// Given row index r, pivs[r] yields the column with low row index r
	auto pivs = vector< optional< entry_t > >(R.dim().first, std::nullopt); // See reference above 
	
	// Reduction algorithm
	for (size_t j; b != e; ++b){
		j = *b; 
		optional< entry_t > piv_i = std::nullopt; // (col index, row value)
		optional< entry_t > low_j = std::nullopt; // (row index, row value)
		while((low_j = R.low(j)) && (piv_i = pivs.at(low_j->first))){ // 
			const size_t i = piv_i->first; // earlier column i of R must already have low row index low_j->first 
			const field_t lambda = low_j->second / piv_i->second;
			R.iadd_scaled_col(j, i, -lambda); 	// zeros pivot in column j
			V.iadd_scaled_col(j, i, -lambda);   // col(j) <- col(j) + s*col(i)
			++reduction_stats[0]; // Keep track of column operations
			f();
		}
    
    // Store pivot entry s.t. pivs[r] yields the column with low row index r
    if (low_j){
    	pivs.at(low_j->first) = make_optional(make_pair(j, low_j->second));
    }
	}
	return; 
}


// Vineyards framework for square matrices
template< PermutableMatrix Matrix, typename Iter, typename Lambda >
void transpose_schedule_full(Matrix& R, Matrix& V, Iter sb, const Iter se, Lambda f){
	using field_t = typename Matrix::value_type;
	using entry_t = typename Matrix::entry_t;
	const size_t nc = R.dim().first;
	const size_t nr = R.dim().second;
	if (nc == 0 || nr == 0){ return; }
	if (nc != nr){ throw std::invalid_argument("R must be square."); }

	// Perform the transpositions
	field_t s = 0;
	size_t status = 0;
	size_t line = 1;
	optional< entry_t > piv_i, piv_j;
	for (size_t i = *sb; sb != se; ++sb, i = *sb){
		auto j = i + 1;
		if (R.column_empty(i) && R.column_empty(j)){
			if (V(i,j) != 0){ V.cancel_lowest(j,i); } // cancel lowest of j using lowest entry in i
			if ((piv_i = R.search_low(i)) && (piv_j = R.search_low(j)) && R(i, piv_j->first)){
				size_t k = piv_i->first, l = piv_j->first; // column indices
				status = k < l ? 1 : 2; 
				if (status == 1){ // Cases 1.1.1
					s = R.cancel_lowest(l,k); R.swap(i,j);
					V.iadd_scaled_col(l,k,s); V.swap(i,j);
				} else {          // Cases 1.1.2
					s = R.cancel_lowest(k,l); R.swap(i,j);
					V.iadd_scaled_col(k,l,s); V.swap(i,j);
				}
			} else {
				R.swap(i,j); V.swap(i,j);
				status = 3; // Case 1.2
			}
		} else if (!R.column_empty(i) && !R.column_empty(j)){
			if (V(i,j) != 0){ // Case 2.1
				if (R.low_index(i) < R.low_index(j)){
					s = R.cancel_lowest(j,i); R.swap(i,j);
					V.iadd_scaled_col(j,i,s); V.swap(i,j);
					status = 4; // Case 2.1.1
				} else {
					s = R.cancel_lowest(j,i); R.swap(i,j); 
					V.iadd_scaled_col(j,i,s); V.swap(i,j); 
					s = R.cancel_lowest(j,i);
					V.iadd_scaled_col(j,i,s);
					status = 5; // Case 2.1.2
				}
			} else {
				R.swap(i,j); V.swap(i,j);
				status = 6; // Case 2.2
			}
		} else if (!R.column_empty(i) && R.column_empty(j)){
			if (V(i,j) != 0){
				s = R.cancel_lowest(j,i); R.swap(i,j); 
				V.iadd_scaled_col(j,i,s); V.swap(i,j);
				s = R.cancel_lowest(j,i);
				V.iadd_scaled_col(j,i,s);
				status = 7; // Case 3.1
			} else {
				R.swap(i,j); V.swap(i,j);
				status = 8; // Case 3.2
			}
		} else {
			status = 9; // Case 4
			if (V(i,j) != 0){ V.cancel_lowest(j,i); }
			R.swap(i,j); V.swap(i,j);
		}
		f(status); // Apply user function
		line++;
	}
}

// Restore right: given column indices i \in [b, e) to restore, apply the donor concept to given indices 
// Postcondition: dr and dv are populated as donor columns
template< PermutableMatrix Matrix, typename Iter >
auto restore_right(Matrix& R, Matrix& V, Iter b, const Iter e) -> std::optional< pair< Matrix, Matrix > >{ 
	using entry_t = typename Matrix::entry_t;
	const size_t ne = std::distance(b, e); 
	if (ne == 0){ return std::nullopt; }

	py::print("restoring");
	int d_low_index = R.low_index(*b), d_low_index_new = 0;
	Matrix dr = R.column(*b);
	Matrix dv = V.column(*b);
	Matrix dr_new = Matrix(dr);
	Matrix dv_new = Matrix(dv); 
	py::print("dr: \n", dr.m);
	py::print("dv: \n", dv.m);
	for (b = std::next(b); b != e; ++b){
		size_t k = *b;
		d_low_index_new = R.low_index(k);

		py::print("k: ", k);
		
		// Condition when to replace donor columns: to be used to save a copy 
		const bool overwrite_donors = R.low_index(k) < d_low_index;
		
		// Determine if columns will need to be saved
		if (overwrite_donors){
			dr_new = R.column(k);
			dv_new = V.column(k);
			py::print("new donor R: \n", dr_new.m);
			py::print("new donor V: \n", dv_new.m);
		}
		
		// Do the reductions 
		auto low_k = R.low(k);
		auto low_donor = dr.low(0);
		const auto s = !low_k ? 1.0 : -(low_k->second/low_donor->second);
		R.m.col(k) = R.m.col(k) + s*dr.m;
		V.m.col(k) = V.m.col(k) + s*dv.m;
		
		py::print("col R[k]: \n", R.column(k).m);
		py::print("col V[k]: \n", V.column(k).m);

		// Save the new donor columns if needed
		if (overwrite_donors){
			d_low_index = d_low_index_new;
			dr = dr_new;
			dv = dv_new;
		}
	}
	// if constexpr (donors){ return std::make_pair(dr, dv); }
	return std::make_optional(std::make_pair(dr, dv));
}

template< PermutableMatrix Matrix, typename Iter, typename Lambda >
void move_schedule_full(Matrix& R, Matrix& V, Iter sb, const Iter se, Lambda f){
	using entry_t = typename Matrix::entry_t;
	const size_t nc = R.dim().first;
	const size_t nr = R.dim().second;
	if (nc == 0 || nr == 0){ return; }
	if (nc != nr){ throw std::invalid_argument("R must be square."); }
	if (std::distance(sb,se) < 2 || std::distance(sb,se) % 2 != 0){  throw std::invalid_argument("Pairs of indices must be passed."); }

	// auto dr = Matrix(R.n_rows(), 1);
	// auto dv = Matrix(V.n_rows(), 1);
	for (size_t i, j; sb != se; sb += 2){
		i = *sb, j = *(sb+1);
		if (i == j){ continue; }
		if (i > j || i >= (nc-1) || j >= nc){  throw std::invalid_argument("Invalid pairs (i,j) passed."); }

		// Collect indices I
		auto I = vector< size_t >();
		for (size_t c = i; c <= j; ++c){
			if (!equals_zero(V(i, c))){
				I.push_back(c);
			}
		}
		py::print(I);

		// Collect indices J
		auto J = vector< size_t >();
		for (size_t c = 0; c < nc; ++c){
			auto low_idx = R.low_index(c);
			if (low_idx >= static_cast<int>(i) && low_idx <= static_cast<int>(j) && R(i,c) != 0){
				J.push_back(c);
			}
		}
		py::print(J);

		// Restore invariants
		auto donors = restore_right(R, V, I.begin(), I.end());
		if (!J.empty()){ restore_right(R, V, J.begin(), J.end()); }
		
 		// Apply permutations
		std::vector< size_t > p = move_right_permutation(i,j,nc);
		py::print(p);
		std::span< size_t > p_span(p);
		R.permute_cols(p_span);
		V.permute(p_span);
		
		// Perform donor replacement, if necessary
		if (donors.has_value()){
			auto [dr,dv] = *donors;
			dr.permute_rows(p_span);
			dv.permute_rows(p_span);
			py::print("dr: \n", dr.m);
			py::print("dv: \n", dv.m);
			R.m.col(j) = dr.m;
			V.m.col(j) = dv.m;
		}	
	}
}