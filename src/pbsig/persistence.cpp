#include "reduction.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Sparse>

namespace py = pybind11;

// Define the field operation and specialize
// https://stackoverflow.com/questions/1596668/logical-xor-operator-in-c
struct AddMod2 { 
	constexpr bool operator()(bool a, bool b) { 
    return(!a != !b); 
  }
};

using FloatSparse = Eigen::SparseMatrix< float >;
using ColumnVector = Eigen::Matrix< float, Eigen::Dynamic, 1>;


// Reducible spec
// { a.dim() } -> std::same_as< pair< size_t, size_t > >;
// { a.low(size_t(0)) } -> std::same_as< optional< pair< size_t, F > > >;
// { a.low_index(size_t(0)) } -> std::same_as< optional< size_t > >;
// { a.low_value(size_t(0)) } -> std::same_as< optional< F > >;
// { a.clear_column(size_t(0)) } -> std::same_as< void >;

// Permutable spec
// { a.swap_rows(size_t(0), size_t(0)) } -> std::same_as< void >;
// { a.swap_cols(size_t(0), size_t(0)) } -> std::same_as< void >;
// { a.swap(size_t(0), size_t(0)) } -> std::same_as< void >;
// { a.permute_rows(std::span< size_t >()) } -> std::same_as< void >;
// { a.permute_cols(std::span< size_t >()) } -> std::same_as< void >;
// { a.column_empty(size_t(0)) } -> std::same_as< bool >;
// { a(size_t(0), size_t(0)) } -> std::same_as< F >; 
using std::vector; 

struct FloatMatrix {
	using value_type = float; 
  using entry_t = std::pair< size_t, value_type >;
  using F = value_type;  // alias 
	FloatSparse m;
	
	constexpr size_t n_rows() const { return m.rows(); };
	constexpr size_t n_cols() const { return m.cols(); };
	
  FloatMatrix(FloatSparse& pm) : m(pm) {};
  
  // FloatMatrix(FloatMatrix const& m) = default;
  // FloatMatrix& operator=(FloatMatrix m) = default;
  FloatMatrix(const size_t nr, const size_t nc) : m(FloatSparse(nr,nc)){ }

	FloatMatrix(vector< size_t > I, vector< size_t > J, vector< float > x, const size_t nr, const size_t nc){
    using triplet_t = Eigen::Triplet< float >;
    vector< triplet_t > nonzeros;
    nonzeros.reserve(I.size());
    for (size_t i = 0; i < I.size(); ++i){
      nonzeros.push_back({ I[i], J[i], x[i] });
    }
    m = FloatSparse(nr,nc);
    m.setFromTriplets(nonzeros.begin(), nonzeros.end());
  };

  auto column(size_t j){
    FloatSparse c = m.col(j).head(n_rows()); // (Eigen::seqN(0, n_rows()),j);
    return FloatMatrix(c);
  }
	
  // -- Interface to make the matrix addable --
  // { a.scale_col(size_t(0), F(0)) } -> std::same_as< void >;
  auto scale_col(size_t j, F val) -> void {
    m.col(j) *= val;
  }

	// { a.iadd_cols(size_t(0), size_t(0)) } -> std::same_as< void >;
	void iadd_cols(size_t i, size_t j){ 
    m.col(i) += m.col(j);
  }

  // iadd_scaled_col(i,j,s) <=> col(i) <- col(i) + s*col(j)
  // { a.iadd_scaled_col(size_t(0), size_t(0), F(0)) } -> std::same_as< void >;
	auto iadd_scaled_col(size_t i, size_t j, F s) -> void {
    m.col(i) += s*m.col(j);
  }

  // { a.dim() } -> std::same_as< pair< size_t, size_t > >;
	auto dim() -> pair< size_t, size_t > {
		return std::make_pair(n_rows(), n_cols());
	} 

	// Returns the lowest entry of column j 
	// { a.low(size_t(0)) } -> std::same_as< optional< pair< size_t, F > > >;
	auto low(size_t j) -> optional< pair< size_t, F > > { 
		if (j >= n_cols()){ throw std::invalid_argument("column index out of range"); }
    using Eigen::placeholders::lastN;
    // std::make_optional()
    auto col_j = m.col(j);
    FloatSparse::InnerIterator it(m,j);
    int i = -1; 
    float x = 0.0;  
    for (; it; ++it){
      if (!equals_zero(it.value())){
        i = it.row();
        x = it.value();
      }
    }
    return i == -1 ? std::nullopt : std::make_optional(std::pair< size_t, value_type >{ i, x });
	}

	// { a.low_index(size_t(0)) } -> std::same_as< int >;
	auto low_index(size_t j) -> int {
		auto le = low(j);
		return(le ? le->first : -1);
	}
	
	// { a.low_value(size_t(0)) } -> std::same_as< optional< F > >;
	auto low_value(size_t j) -> F {
		auto le = low(j);
		return(le ? le->second : 0);
	}

	// { a.clear_column(size_t(0)) } -> std::same_as< void >;
	auto clear_column(size_t j) -> void {
    m.col(j) *= 0;
  }

  // Zero-out the lowest non-zero of j using the lowest non-zero of i, if both exist (and its possible)
	// { a.cancel_lowest(size_t(0), size_t(0)) } -> std::same_as< F >;
	auto cancel_lowest(size_t j, size_t i) -> F {
		auto low_i = low(i); // row index, value 
		auto low_j = low(j); // row index, value 
		if (low_i && low_j && low_i->first == low_j->first){
      const auto s = -(low_j->second/low_i->second);
      iadd_scaled_col(j, i, s); // col(j) <- col(j) + s*col(i)
      return s;
		}
    return 0;
	}

  // -- Interface to make the matrix permutable --
	
	// { a.swap_rows(size_t(0), size_t(0)) } -> std::same_as< void >;
	void swap_rows(size_t i, size_t j){ 
    Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P(n_rows());
    P.setIdentity();
    P.indices()[i] = j;
    P.indices()[j] = i;
    m = P * m;
  }
	
	// { a.swap_cols(size_t(0), size_t(0)) } -> std::same_as< void >;
	void swap_cols(size_t i, size_t j){ 
    Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P(n_rows());
    P.setIdentity();
    P.indices()[i] = j;
    P.indices()[j] = i;
    m = m * P;
  }
	
	// { a.swap(size_t(0), size_t(0)) } -> std::same_as< void >;
	void swap(size_t i, size_t j){
    Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P(n_rows());
    P.setIdentity();
    P.indices()[i] = j;
    P.indices()[j] = i;
    m = P * m * P;
  }
	
	// { a.permute_rows(std::span< size_t >()) } -> std::same_as< void >;
	void permute_rows(std::span< size_t > ind){ 
    Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P(n_rows());
    P.setIdentity();
    for (size_t i = 0; i < ind.size(); ++i){
      P.indices()[i] = ind[i];
    }
    m = P * m;
  }
	
	// { a.permute_cols(std::span< size_t >()) } -> std::same_as< void >;
	void permute_cols(std::span< size_t > ind){ 
    Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P(n_rows());
    P.setIdentity();
    for (size_t i = 0; i < ind.size(); ++i){
      P.indices()[i] = ind[i];
    }
    m = m * P;
  }
	
	void permute(std::span< size_t > ind){ 
    Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P(n_rows());
    P.setIdentity();
    for (size_t i = 0; i < ind.size(); ++i){
      P.indices()[i] = ind[i];
    }
    m = P * m * P;
  }
	// { a.column_empty(size_t(0)) } -> std::same_as< bool >;
	bool column_empty(size_t j){  
    return low(j).has_value();
  }
  
  // { a.operator()(size_t(0), size_t(0)) } -> std::same_as< F >;
  auto operator()(size_t i, size_t j) const -> F {
    return m.coeff(i,j);
  }
	
	// template <typename ... Args>
	// void row(Args&& ... args){ m.row(std::forward<Args>(args)...); }
	

	// template <typename ... Args>
	// void column(Args&& ... args){ m.column(std::forward<Args>(args)...); }
	
	// template <typename ... Args>
	// void write_column(Args&& ... args){ m.write_column(std::forward<Args>(args)...); }
	
	// template <typename ... Args>
	// auto find_in_col(Args&& ... args) -> std::optional< pair< size_t, F > >{ 
	// 	return m.find_in_col(std::forward<Args>(args)...); 
	// }
		
	// template <typename ... Args>
	// void add_cols(Args&& ... args){ m.add_cols(std::forward<Args>(args)...); }
	
	// // unneeded
	// template <typename ... Args>
	// void scale_col(Args&& ... args){ }
	
	// void add_scaled_col(size_t i, size_t j, size_t k, F lambda = true){
	// 	if (i != k && j != k){ throw std::invalid_argument("i or j must equal k."); }
	// 	add_cols(i,j,k);
	// }
	
	// template <typename ... Args>
	// F operator()(Args&& ... args){ return m(std::forward<Args>(args)...); }
	
	// template <typename ... Args>
	// void assign_column(Args&& ... args){
	// 	m.assign_column(std::forward<Args>(args)...);
	// }

	// Given column j with low(j) = r, find a column 'i' satisfying i < j and low(i) = low(j) = r
	// { a.find_low(size_t(0), size_t(0)) } -> std::same_as< std::optional< pair< size_t, F > > >; 
	auto find_low(size_t j, size_t r) -> std::optional< pair< size_t, F > >  {
		for (size_t i = 0; i < j; ++i){
			auto low_i = low(i);
			if (low_i && low_i->first == r){
				return(make_optional(make_pair(i, low_i->second)));	// Note this is (column index, field value), not (row index, field value)
			}
		}
		return(std::nullopt);
	}

  // Searches the low entries for a column 'k' satisfying low(k) = i
	// { a.find_low(size_t(0), size_t(0)) } -> std::same_as< std::optional< pair< size_t, F > > >; 
	auto search_low(size_t i) -> std::optional< pair< size_t, F > >  {
		for (size_t k = 0; k < n_cols(); ++k){
			auto low_k = low(k);
			if (low_k && low_k->first == i){
				return(make_optional(make_pair(k, low_k->second)));	// Note this is (column index, field value), not (row index, field value)
			}
		}
		return(std::nullopt);
	}
};

// TODO: The references don't matter here. Either accept and return triplet form, or 
// re-make the types to populate the boundary matrix columns implicitly like PHAT
auto phcol(Eigen::SparseMatrix< float > R_, Eigen::SparseMatrix< float > V_, std::vector< size_t > I) -> std::pair< FloatSparse, FloatSparse > {
  auto R = FloatMatrix(R_);
  auto V = FloatMatrix(V_);
  size_t ii = 0;
  pHcol(R, V, I.begin(), I.end(), [&ii](){
    if ((ii % 100) == 0 && PyErr_CheckSignals() != 0){ throw py::error_already_set(); }
    ii++;
  });
  const auto filter_zeros = [](const auto& row, const auto& col, const auto& value) -> bool {
    return !equals_zero(value);
  };
  R.m.prune(filter_zeros);
  V.m.prune(filter_zeros);
  return std::make_pair(R.m, V.m);
}

auto reduction_stats(bool reset = false) -> size_t {
  if (reset){
    _reduction_stats[0] = 0;
  }
  return static_cast< size_t >(_reduction_stats[0]);
}

auto move_right(Eigen::SparseMatrix< float > R_, Eigen::SparseMatrix< float > V_, size_t i, size_t j) -> std::pair< FloatSparse, FloatSparse > {
  auto R = FloatMatrix(R_);
  auto V = FloatMatrix(V_);
  std::array< size_t, 2 > p = { i, j };
  move_schedule_full(R, V, p.begin(), p.end(), [](){
    return; 
  });
  const auto filter_zeros = [](const auto& row, const auto& col, const auto& value) -> bool {
    return !equals_zero(value);
  };
  R.m.prune(filter_zeros);
  V.m.prune(filter_zeros);
  return std::make_pair(R.m, V.m);
}

// pip install --no-deps --no-build-isolation --editable .
// /usr/local/Cellar/llvm/12.0.1/bin/clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -fPIC -O2 -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/pbsig/extern/eigen -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/lib/python3.9/site-packages/pybind11/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -c src/pbsig/persistence.cpp -o build/temp.macosx-10.9-x86_64-3.9/src/pbsig/persistence.o -fvisibility=hidden -g0 -stdlib=libc++ -std=c++1 -mmacosx-version-min=10.9 -fvisibility=hidden -g0 -stdlib=libc++ -std=c++1 -mmacosx-version-min=10.9 -fvisibility=hidden -g0 -stdlib=libc++ -std=c++1 -mmacosx-version-min=10.9 -fvisibility=hidden -g0 -stdlib=libc++ -std=c++1 -mmacosx-version-min=10.9 -fvisibility=hidden -g0 -stdlib=libc++ -std=c++1 -mmacosx-version-min=10.9 -fvisibility=hidden -g0 -stdlib=libc++ -std=c++1 -mmacosx-version-min=10.9 -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -arch x86_64 -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -fPIC -O2 -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -arch x86_64 -std=c++20 -Wall -Wextra -march=native -O3 -fopenmp
PYBIND11_MODULE(_persistence, m) {
  m.doc() = "persistence module";
  m.def("phcol", &phcol);
  m.def("move_right", &move_right);
  m.def("reduction_stats", &reduction_stats);
  // m.def("test_low", &test_sparse_matrix);
  
  py::class_< FloatMatrix >(m, "FloatMatrix")
   .def(py::init< FloatSparse& >())
   .def(py::init< vector< size_t >, vector< size_t >, vector< float >, size_t, size_t >())
   .def_property_readonly("n_rows", &FloatMatrix::n_rows)
   .def_property_readonly("n_cols", &FloatMatrix::n_cols)
   .def_property_readonly("dim", &FloatMatrix::dim)
   .def("low", [](FloatMatrix& A, size_t j) -> int {
      auto piv = A.low(j);
      return piv ? piv->first : -1;
   })
   .def("clear_column", &FloatMatrix::clear_column)
   .def("iadd_scaled_col", &FloatMatrix::iadd_scaled_col)
   .def("swap_rows", &FloatMatrix::swap_rows)
   .def("swap_cols", &FloatMatrix::swap_cols)
   .def("swap", &FloatMatrix::swap)
   .def("permute_rows", [](FloatMatrix& m, vector< size_t > p){
      std::span< size_t > p_span(p);
   })
   .def("permute_cols", &FloatMatrix::permute_cols)
   .def("permute", &FloatMatrix::permute)
   .def("as_spmatrix", [](FloatMatrix& M) -> FloatSparse {
      return M.m;
   })
   ;
}