#include <vector>
#include <algorithm>
#include <concepts>
#include <iterator>
#include <cstddef>
#include <array>
#include <optional>
#include <tuple>
#include <sstream>

#include "combinadic.h"

using std::size_t;
using std::tuple;
using std::pair; 
using std::vector; 
using std::unique_ptr;
using std::array; 
using std::optional;
using std::make_tuple;
using std::make_pair;
using std::make_optional;

// Sparse Matrix representation that is robust to row/column permutations 
template< typename T, class BinaryOperation >
class PspMatrix {
public: 
	using entry_t = pair< size_t, T >;
	using column_t = vector< entry_t >;
	using entries_ptr = unique_ptr< vector< entry_t > >;
	using value_type = T;
	
	vector< entries_ptr > columns; // non-zero entries
	vector< size_t > cto;  // current-to-original map
	vector< size_t > otc;	 // original-to-current map
	
	BinaryOperation add; 
	array< size_t, 2 > size = { 0, 0 };
	size_t nnz = 0;
	constexpr size_t n_rows() const { return size[0]; };
	constexpr size_t n_cols() const { return size[1]; };
	
	// Constructor only allocate empty matrix
	PspMatrix(const size_t m, const size_t n) : columns(n), cto(m), otc(m), size({ m, n }){
		for (size_t i = 0; i < n; ++i){
			columns[i] = std::make_unique< vector< entry_t > >();
		}
		std::iota(std::begin(cto), std::end(cto), 0);
		std::iota(std::begin(otc), std::end(otc), 0);
	}; 
	
	// Rule of Five
	PspMatrix(PspMatrix&&) = default; 									// move 
	PspMatrix& operator=(PspMatrix&&) = default;        // move assign
	PspMatrix(const PspMatrix&) = default;							// copy 
	PspMatrix& operator=(const PspMatrix&) = default;   // copy assign  
	~PspMatrix() = default; 														// default destructor 
	
	template< typename TripletIt, typename = decltype(*std::declval<TripletIt&>(), void(), ++std::declval<TripletIt&>(), void()) >
	void initialize(TripletIt b, const TripletIt e){
		using triplet_t = tuple< size_t, size_t, T >;
		using iter_t = typename std::iterator_traits< TripletIt >::value_type;
		static_assert(std::is_same< triplet_t, iter_t >::value, "Invalid triplet specification.");
		T val; 
		for (size_t i, j; b != e; ++b){
			std::tie(i,j,val) = *b;
			this->insert(i,j,val);
		}
	}
	
	// Tests  whether value_type is 0
	static bool equals_zero(value_type val) {
		if constexpr (std::is_integral_v< value_type >){
			return val == 0; 
		} else {
			return std::abs(val) <= std::numeric_limits< value_type >::epsilon();
		}
	}
	
	// Alternative triple-construction method 
	template< typename TripletIt >
	PspMatrix(TripletIt b, const TripletIt e, const size_t m, const size_t n) : PspMatrix(m,n) {
		initialize(b, e);
	}
	
	// Non-iterator construction 
	void construct(vector< size_t > i, vector< size_t > j, vector< T > x){
		// if (i.size() != j.size() || j.size() != x.size()){ Rcpp::stop("Invalid input"); }
		// int max_row = *std::max_element(i.begin(), i.end());
		// int max_col = *std::max_element(j.begin(), j.end());
		// if (max_row >= m->n_rows() || max_col >= m->n_cols()){ Rcpp::stop("Invalid input"); }
		auto nz_entries = vector< std::tuple< size_t, size_t, T > >();
		for (size_t pos = 0; pos < x.size(); ++pos){
			nz_entries.push_back(std::make_tuple(i[pos], j[pos], x[pos]));
		}
		this->initialize(nz_entries.begin(), nz_entries.end());
	}
	
	// Apply an lambda function to each non-zero element of the matrix
	template< typename Lambda >
	void apply(const Lambda& f){
		using output_t =  typename std::result_of< Lambda(size_t, size_t, T) >::type;
		static constexpr bool is_void = std::is_same< output_t, void >::value;
		static constexpr bool is_T = std::is_same< output_t, T >::value;
		static_assert(is_void || is_T, "Result type must be void or T.");
		for (size_t j = 0; j < n_cols(); ++j){ apply_col(j, f); }
	}
	
	template< typename Lambda >
	void apply_col(const size_t j, const Lambda& f){
		using output_t =  typename std::result_of< Lambda(size_t, size_t, T) >::type;
		static constexpr bool is_void = std::is_same< output_t, void >::value;
		static constexpr bool is_T = std::is_same< output_t, T >::value;
		static_assert(is_void || is_T, "Result type must be void or T.");
		if (columns[j]){
			vector< entry_t >& v = *columns[j];
			for (auto& e: v){
				if constexpr (is_void){
					f(otc[e.first], j, e.second);
				} else {
					T ne = f(otc[e.first], j, e.second);
					if (ne != e.second){ e.second = ne; }
				}
			}
		}
	}

	// Applies given 'f' to non-zero entries of column 'c'. 
	// Calls 'f' with signature f(< row index >, < non-zero entry >)
	template < typename Lambda >
	void column(size_t c, Lambda&& f){
		if (c >= n_cols()){ return; }
		vector< entry_t >& v = *columns[c]; 
		for (entry_t& e: v){
			f(otc[e.first], e.second);
		}
	}
	
	// struct col_iterator {
	// 	
	// };
	
	std::optional< entry_t > find_in_col(const size_t j, const size_t r){
		if (j >= n_cols()){ throw std::invalid_argument("Invalid column index given."); }
		if (r >= n_rows()){ throw std::invalid_argument("Invalid row index given."); }
		vector< entry_t >& entries = *columns.at(j);
		const auto ri = cto[r];
		auto el = std::lower_bound(std::begin(entries), std::end(entries), ri, [](entry_t& e, size_t index){
			return(e.first < index);
		});
		if (el == std::end(entries) || el->first != ri){ return std::nullopt; } 
		else {
			return(std::make_optional(std::make_pair(otc[el->first], el->second)));
		}
	}
	
	template < typename OutputIter >
	void write_column(size_t c, OutputIter out){
		if (c >= n_cols()){ return; }
		if (!columns[c]){ columns[c] = std::make_unique< vector< entry_t > >(); }; 
		vector< entry_t >& col = *columns[c];
		std::transform(col.begin(), col.end(), out, [&](const entry_t& el){
			return(std::make_pair(otc[el.first], el.second));
		});
	}
	
	template < typename Iter >
	void assign_column(size_t c, Iter b, const Iter e){
		static_assert(std::is_same_v<typename std::iterator_traits< Iter >::value_type, entry_t >, "Incorrect value_type for iterator");
		if (c >= n_cols()){ throw std::invalid_argument("Invalid column index"); }
		
		// Re-map the current indices to the original indices to maintain consistency with interface 
		// mke copy here
		vector< entry_t > nc(b, e); 
		std::transform(nc.begin(), nc.end(), nc.begin(), [&](entry_t& el){ return make_pair(cto[el.first], el.second); });
		
		// Apply the assignment
		if (!bool(columns.at(c))){ 
			columns.at(c) = std::make_unique< vector< entry_t > >(nc.begin(),nc.end()); 
		} else {
			columns[c]->clear();
			columns[c]->assign(nc.begin(),nc.end());
		}
	}
	
	// Returns whether a column is completely zero
	bool column_empty(size_t j) const {
		if (j >= n_cols()){ 
			std::stringstream ss;
			ss << "Invalid column index " << j << " chosen.";
			std::string bad_msg = ss.str();
			throw std::invalid_argument(bad_msg); 
		}
		if (!bool(columns[j]) || columns[j]->size() == 0){ return(true); }
		bool col_empty = std::all_of(columns[j]->begin(), columns[j]->end(), [](auto entr){
			return(equals_zero(entr.second));
		});
		return(col_empty);
	}
	
	// Applies given 'f' to non-zero entries of row 'r'. 
	// Calls 'f' with signature f(< column index >, < non-zero entry >)
	template < typename Lambda >
	void row(size_t r, Lambda&& f){
		if (r >= n_rows()){ return; }
		for (size_t j = 0; j < n_cols(); ++j){
			vector< entry_t >& v = *columns[j];
			// TODO: fix this
			// Search for the a non-zero entry in row 'r' in O(log(n)) time 
			auto el = std::lower_bound(std::begin(v), std::end(v), cto[r], [this](entry_t& e, size_t index){
				return(e.first < index);
			});
			// If not found, continue to the next column 
			if (el == std::end(v) || otc[el->first] != r){
				continue; 
			} else {
				f(j, el->second);
			}
		}
	}
	
	// Removes entries below a certain threshold from the matrix
	// TODO: Fix this
	void clean(const T threshold){
		auto to_remove = vector< pair< size_t, size_t > > (); 
		apply([threshold, &to_remove](size_t i, size_t j, T val){
			if (val <= threshold){
				to_remove.push_back(std::make_pair(i,j));
			}
		});
		for (auto p: to_remove){ remove(p.first, p.second); }
	}
	
	// Removes the (i,j)th entry of the matrix, if it exists
	// TODO: Fix this
	void remove(size_t i, size_t j){
		if (i >= n_rows() || j >= n_cols()){ return; }
		if (!bool(columns.at(j))){ return; }
		vector< entry_t >& entries = *columns.at(j);
		
		// auto el = std::lower_bound(std::begin(entries), std::end(entries), cto[i], [this](entry_t& e, size_t index){
		// 	return(e.first < index);
		// });
		auto o_idx = cto[i]; // original index to search for
		auto el = std::lower_bound(std::begin(entries), std::end(entries), o_idx, [](entry_t& e, size_t index){
			return(e.first < index);
		});
		// auto el = std::find_if(std::begin(entries), std::end(entries), [this, i](entry_t& e){
		// 	return(otc[e.first] == i);
		// });
		// if (el == std::end(entries) || otc[el->first] != i){
		if (el == std::end(entries) || el->first != o_idx){
			return; 
		} else {
			entries.erase(el);
			--nnz;
		}
	}
	
	// Adds column entries in given iterator directly to column k
	template< typename Iter >
	void add_col(Iter b, const Iter e, size_t k){
		if (k >= n_cols()){ throw std::invalid_argument("Invalid indexes given"); }
		if (!bool(columns.at(k))){ columns.at(k) = std::make_unique< vector< entry_t > >(); }
				
		// Re-map the current indices to the original indices to maintain consistency with interface 
		vector< entry_t > col_copy;
		col_copy.reserve(std::distance(b,e));
		std::transform(b, e, std::back_inserter(col_copy), [&](entry_t& el){ return make_pair(cto[el.first], el.second); });
		
		auto first1 = columns[k]->begin(), last1 = columns[k]->end(); // target column
		auto first2 = col_copy.begin(), last2 = col_copy.end();
		vector< entry_t > to_add; 
		auto to_remove = vector< pair< size_t, size_t > > (); 
		while(true){
			if (first1 == last1){ 
				nnz += std::distance(first2, last2);
				std::copy(first2, last2, std::back_inserter(*columns[k])); 
				break;
			}
			if (first2 == last2){ break; }
			if ((*first1).first == (*first2).first){
				(*first1).second = add((*first1).second, (*first2).second);
				
				if (equals_zero(first1->second)){ // target column 
					to_remove.push_back(std::make_pair(first1->first,k));
				}

				++first1; ++first2;
			} else if ((*first1).first < (*first2).first){
				++first1;
			} else if ((*first2).first < (*first1).first){
				if (!equals_zero(first2->second)){
					to_add.push_back(*first2);
				}
				++first2;
			} else {
				throw std::logic_error("Invalid case");
			}
		}
		// Add entries retroactively that would've invalidated iterators above
		for (auto e: to_add){
			insert(otc[e.first], k, e.second);
		}
		for (auto p: to_remove){ remove(p.first, p.second); }
	}
	
	// Performs the column additions ( i + j -> k ) where i,j,k specify column indices, k == i or k == j. 
	// Currently k must equal either i or j
	void add_cols(size_t i, size_t j, size_t k){
		if (i >= n_cols() || j >= n_cols()){ throw std::invalid_argument("Invalid indexes given"); }
		if (k != i && k != j){ throw std::invalid_argument("target column must be one of the given columns."); }
		
		// Decide on source and target columns
		const size_t s = i == k ? j : i; 
		const size_t t = k;
		if (!bool(columns.at(s))){ columns.at(s) = std::make_unique< vector< entry_t > >(); }
		if (!bool(columns.at(t))){ columns.at(t) = std::make_unique< vector< entry_t > >(); }
		
		auto first1 = columns[t]->begin(), last1 = columns[t]->end();
		auto first2 = columns[s]->begin(), last2 = columns[s]->end();
		vector< entry_t > to_add; 
		while(true){
			if (first1 == last1){ 
				nnz += std::distance(first2, last2);
				std::copy(first2, last2, std::back_inserter(*columns[t])); 
				break;
			}
			if (first2 == last2){ break; }
			if ((*first1).first == (*first2).first){
				(*first1).second = add((*first1).second, (*first2).second);
				++first1; ++first2;
			} else if ((*first1).first < (*first2).first){
				++first1;
			} else if ((*first2).first < (*first1).first){
				to_add.push_back((*first2));
				++first2;
			} else {
				throw std::logic_error("Invalid case");
			}
		}
		// Add entries retroactively that would've invalidated iterators above
		for (auto e: to_add){
			insert(otc[e.first], t, e.second);
		}
	}	
	
	// Performs the operation: row(i) = row(i) + row(j)  
	void add_rows(size_t i, size_t j){
		if (i >= n_rows() || j >= n_rows()){ throw std::invalid_argument("Invalid"); }
				
		// Collect the rows 
		auto row_i = vector< pair< size_t, T > >();
		auto row_j = vector< pair< size_t, T > >();
		row(i, [&row_i](auto c, auto e){ row_i.push_back(make_pair(c, e)); });
		row(j, [&row_j](auto c, auto e){ row_j.push_back(make_pair(c, e)); });
		
		auto first1 = row_i.begin(), last1 = row_i.end(); 
		auto first2 = row_j.begin(), last2 = row_j.end(); 
		
		auto to_add = vector< tuple< size_t, size_t, T > >(); 
		while(true){
			if (first1 == last1){
				nnz += std::distance(first2, last2);
				for (; first2 != last2; ++first2){
					to_add.push_back(make_tuple(i, (*first2).first, (*first2).second));
				}
				break;
			}
			if (first2 == last2){ break; }
			if ((*first1).first == (*first2).first){
				(*first1).second = add((*first1).second, (*first2).second);
				++first1;
				++first2;
			} else if ((*first1).first < (*first2).first){
				++first1;
			} else if ((*first2).first < (*first1).first){
				to_add.push_back(make_tuple(i, (*first2).first, (*first2).second));
				++first2;
			} else {
				throw std::logic_error("Invalid case");
			}
		}
		
		for (auto te: to_add){
			insert(otc[ std::get<0>(te) ], std::get<1>(te), std::get<2>(te));
		}
		
	}
	
	// Returns an optional containing the lowest-nonzero entry of column j
	optional< pair< size_t, value_type > > lowest_nonzero(const size_t j) const {
		// if column is empty, return nullopt
		// if (column_empty(j)){ return(std::nullopt); }
		if (j >= n_cols()){ throw std::invalid_argument("Invalid column index"); }
		if (!bool(columns[j]) || columns[j]->size() == 0){ return(std::nullopt); }
		
		// Get non-zero entry w/ maximum row index
		// Note this essentially has to be take O(nnz_j) time, since we are asking for the lowest 
		// *current* row index instead of the lowest *original* index
		const auto& c = *columns[j];
		size_t best = 0; 
		for (size_t ri = 0; ri < c.size(); ++ri){
			if (otc[c[ri].first] >= otc[c[best].first] && !equals_zero(c[ri].second)){
				best = ri; 
			}
		}
		if (equals_zero(c[best].second)){ return(std::nullopt); }
		return std::make_optional(std::make_pair(otc[c[best].first], c[best].second));
	}
	
	void swap_rows(size_t i, size_t j){
		if (i >= n_rows() || j >= n_cols()){ return; }
		const auto i_idx = cto[i], j_idx = cto[j];
		std::swap(cto[i], cto[j]);
		std::swap(otc[i_idx], otc[j_idx]);
	}
	void swap_cols(size_t i, size_t j){
		if (i >= n_cols() || j >= n_cols()){ return; }
		std::swap(columns[i], columns[j]);
	}
	void swap(size_t i, size_t j){
		swap_rows(i,j);
		swap_cols(i,j);
	}
	
	template< typename Iter >
	void permute_rows(Iter b, const Iter e){
		const size_t n = std::distance(b,e);
		if (n != n_rows()){ throw std::invalid_argument("Permutation must match number of rows."); }
		
		vector< size_t > p(b,e);
		apply_permutation(cto.begin(), cto.end(), p.begin());
		otc = inverse_permutation(cto.begin(), cto.end());
		
		// // Prepare permutation + inverse permutation
		// vector< size_t > p(b,e), ip(n);
		// for (size_t i = 0; i < n; ++i){ ip[p[i]] = i; } //   inverse permutation 
		// 
		// // Map the row permutation to a new vector
		// for (size_t i = 0; i < n; ++i){ 
		// 	p[i] = cto[p[i]];    // compute permutation vector 
		// 	ip[i] = cto[ip[i]];	 // compute inverse permutation vector
		// }
		// 
		// // Update the row correspondences
		// std::move(p.begin(), p.end(), cto.begin());
		// std::move(ip.begin(), ip.end(), otc.begin());
	}

	// Convenience method
	void permute_rows(const vector< size_t > p){
		this->permute_rows(p.begin(), p.end());
	}
	
	// Permutes the columns of the matrix according to the permutation given in [b,e)
	template< typename Iter >
	void permute_cols(Iter b, const Iter e){
		const size_t n = std::distance(b,e);
		if (n != n_cols()){ throw std::invalid_argument("Permutation must match number of columns."); }
		vector< size_t > p(b,e);
		apply_permutation(columns.begin(), columns.end(), p.begin());
		// Map the columns pointers to a new vector
		// vector< entries_ptr > p(n);
		// for (size_t i = 0; b != e; ++b, ++i){
		// 	std::swap(p[i], columns[*b]);
		// }	
		// std::move(p.begin(), p.end(), columns.begin());
	}
	void permute_cols(const vector< size_t > p){
		this->permute_cols(p.begin(), p.end());
	}
	
	template< typename Iter >
	void permute(Iter b, const Iter e){
		if (n_rows() != n_cols()){ throw std::invalid_argument("Permute only possible for square matrices."); }
		permute_cols(b,e);
		permute_rows(b,e);
	}
	
	template < typename OutputStream >
	void print(OutputStream& os){
		std::stringstream ss;
		ss << "Sparse matrix (" << std::to_string(n_rows()) << "," << std::to_string(n_cols()) << ")";
		os << ss.str() << std::endl;
		
		for (size_t c = 0; c < n_cols(); ++c){
			vector< entry_t >& v = *columns[c];
			if (v.size() > 0){
				os << std::to_string(c) << "|";
				for (auto& e: v){
					os << "(" << std::to_string(e.first) << ":" << std::to_string(e.second) << ") "; 
				}
				os << std::endl;
			}
		}
	}
	
	// Element access
	T operator()(size_t i, size_t j) const {
		if (i >= n_rows() || j >= n_cols() || !bool(columns.at(j))){ return(static_cast< T >(0)); }
		vector< entry_t >& entries = *columns.at(j);
		auto o_idx = cto[i];
		auto el = std::lower_bound(std::begin(entries), std::end(entries), o_idx, [](entry_t& e, size_t index){
			return(e.first < index);
		});
		if (el == std::end(entries)){ return(static_cast< T >(0)); }
		return(el->first == o_idx ? el->second : 0);
	}
	
	// Sets an element at position (i,j) to value 'val'
	// TODO: Test to ensure lower_bound applies here
	void insert(size_t i, size_t j, T val) {
		if (i >= n_rows() || j >= n_cols()){ throw std::invalid_argument("Invalid"); }
		if (!bool(columns.at(j))){ columns.at(j) = std::make_unique< vector< entry_t > >(); }
		vector< entry_t >& entries = *columns.at(j);
		auto o_idx = cto[i]; // original index to search for
		auto el = std::lower_bound(std::begin(entries), std::end(entries), o_idx, [](entry_t& e, size_t index){
			return(e.first < index);
		});
		if (el == std::end(entries) || el->first != o_idx){
			entries.insert(el, std::make_pair(o_idx, val));
			++nnz;
		} else {
			el->second = val;
		}
	}
};
