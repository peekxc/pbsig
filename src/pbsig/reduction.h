#include <string>
#include <cstddef>
#include <concepts>
#include <iostream>
#include <optional>
#include <tuple>
#include "reduction_concepts.h"

using std::pair; 
using std::make_pair;
using std::size_t; 
using std::optional;
using std::tuple;
using std::make_tuple;
using std::vector; 

// Global static variable to store statistics
// [0] stores number of column operations 
// [1] stores number of field operations, if possible
static std::array< size_t, 2 > reduction_stats; 

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
int restore_right(Matrix& R, Matrix& V, Iter b, const Iter e, Matrix& dr, Matrix& dv){
	using entry_t = typename Matrix::entry_t;
	const size_t ne = std::distance(b, e); 
	if (ne == 0){ return(0); }
	
	int nr = 0; 
	auto d_low_index = R.low_index(*b);
	auto d_low_index_new = std::optional< size_t >{ std::nullopt };
	dr.clear_column(0); 
	dv.clear_column(0);
	R.col(*b) = dr; // need to support column assignment
	V.col(*b) = dv; // need to support column assignment
	
	// Restore columns 
	for (b = std::next(b); b != e; ++b){
		size_t k = *b;
		d_low_index_new = R.low_index(k);
		
		// Condition when to replace donor columns
		const bool overwrite_donors = d_low_index_new.has_value() 
			&& (d_low_index && d_low_index_new.value() < d_low_index.value())
			|| (!d_low_index_new.has_value());
		
		// Determine if columns will need to be saved
		if (overwrite_donors){
			R.col(k) = dr_new;
			V.col(k) = dv_new;
		}
		
		// Do the reductions 
		// R.iadd_cols(dr.begin(), dr.end(), k); // need to fix by adding new concepts, such as cancel lowest with a column 
		// V.iadd_cols(dv.begin(), dv.end(), k);
		++nr;

		// Save the new donor columns if needed
		if (overwrite_donors){
			d_low_index = d_low_index_new;
			dr = dr_new;
			dv = dv_new;
			dr_new.clear_column(0);
			dv_new.clear_column(0);
		}
	}
	return(nr);
}