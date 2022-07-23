// cppimport

#include "PspMatrix.h"
#include "reduction.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;


// Define the field operation and specialize
struct AddFloat { 
	constexpr bool operator()(float a, float b) { return(a + b); }
};
typedef PspMatrix< float, AddFloat > PspFloatMatrix;


// TODO: switch to CRTP-facade like technique like https://vector-of-bool.github.io/2020/06/13/cpp20-iter-facade.html
// or use a type-erasure / duck-typing strategy to remove many of the overloads 
struct FloatMatrix {
	using F = PspFloatMatrix::value_type;
	using value_type = F; 
	using entry_t = PspFloatMatrix::entry_t;
	// using optional< pair< size_t, entry_t > >
	
	PspFloatMatrix& m;
	
	constexpr size_t n_rows() const { return m.n_rows(); };
	constexpr size_t n_cols() const { return m.n_cols(); };
	
	FloatMatrix(PspFloatMatrix& pm) : m(pm) { };
	
	// Returns the lowest entry of column j 
	// { a.low(size_t(0)) } -> std::same_as< optional< pair< size_t, F > > >;
	auto low(size_t j) { 
		if (j >= n_cols()){ throw std::invalid_argument("column index out of range"); }
		return m.lowest_nonzero(j);
	}

	// { a.low_index(size_t(0)) } -> std::same_as< optional< size_t > >;
	auto low_index(size_t j){
		auto le = low(j);
		return(le ? std::make_optional(le->first) : std::nullopt);
	}
	
	// { a.low_value(size_t(0)) } -> std::same_as< optional< F > >;
	auto low_value(size_t j){
		auto le = low(j);
		return(le ? std::make_optional(le->second) : std::nullopt);
	}
	
	// Use column s to cancel lowest entry of t, if it exists
	// { a.cancel_lowest(size_t(0), size_t(0)) } -> std::same_as< void >;
	void cancel_lowest(size_t t, size_t s){
		auto low_t = m.lowest_nonzero(t); // optional< pair< size_t, value_type > >
		auto low_s = m.lowest_nonzero(s);

    vector< entry_t > col_s; 
    vector< entry_t > col_t;

		if (low_t && low_s && low_s->first == low_t->first){
      column(size_t c, [](size_t ri, float v){
        
      });
      const auto lambda = -(low_t->second/low_s->second)
      apply_col(t, [lambda](size_t ri, value_type v){
        return(v + lambda);
      });
      // lambda*low_s->second
			m.add_cols(s, t, t); // (s + t |-> t)
			// Rprintf("added columns %d to %d (s low: %d, t low: %d)\n", s, t, low_s->first, low_t->first);
		}
	}
	
	// Zero's out column j
	auto clear_column(size_t j){
		if (m.columns.at(j)){
			m.columns[j]->clear();
		}
	}

	// { a.dim() } -> std::same_as< pair< size_t, size_t > >;
	auto dim() -> pair< size_t, size_t > {
		return std::make_pair(m.size[0], m.size[1]);
	} 
	
	// Given column j which has low row index 'j_low_index', find the column 'i' which has the same low row index
	// { a.find_low(size_t(0), size_t(0)) } -> std::same_as< std::optional< pair< size_t, F > > >; 
	auto find_low(size_t j, size_t j_low_index) -> std::optional< pair< size_t, F > >  {
		for (size_t i = 0; i < j; ++i){
			auto low_i = low(i);
			if (low_i && low_i->first == j_low_index){
				return(make_optional(make_pair(i, low_i->second)));	// Note this is (column index, field value), not (row index, field value)
			}
		}
		return(std::nullopt);
	}
	
	template <typename ... Args>
	void add_col(Args&& ... args){ m.add_col(std::forward<Args>(args)...); }
	
	// { a.swap_rows(size_t(0), size_t(0)) } -> std::same_as< void >;
	template <typename ... Args>
	void swap_rows(Args&& ... args){ m.swap_rows(std::forward<Args>(args)...); }
	
	// { a.swap_cols(size_t(0), size_t(0)) } -> std::same_as< void >;
	template <typename ... Args>
	void swap_cols(Args&& ... args){ m.swap_cols(std::forward<Args>(args)...); }
	
	// { a.swap(size_t(0), size_t(0)) } -> std::same_as< void >;
	template <typename ... Args>
	void swap(Args&& ... args){ m.swap(std::forward<Args>(args)...); }
	
	// { a.permute_rows(std::span< size_t >()) } -> std::same_as< void >;
	template <typename ... Args>
	void permute_rows(Args&& ... args){ m.permute_rows(std::forward<Args>(args)...); }
	
	// { a.permute_cols(std::span< size_t >()) } -> std::same_as< void >;
	template < typename ... Args >
	void permute_cols(Args&& ... args){ 
		m.permute_cols(std::forward<Args>(args)...);
	}
	
	template <typename ... Args>
	void permute(Args&& ... args){ m.permute(std::forward<Args>(args)...); }
	
	template <typename ... Args>
	void row(Args&& ... args){ m.row(std::forward<Args>(args)...); }
	
	// { a.column_empty(size_t(0)) } -> std::same_as< bool >;
	template <typename ... Args>
	bool column_empty(Args&& ... args){ return m.column_empty(std::forward<Args>(args)...); }
	
	template <typename ... Args>
	void column(Args&& ... args){ m.column(std::forward<Args>(args)...); }
	
	template <typename ... Args>
	void write_column(Args&& ... args){ m.write_column(std::forward<Args>(args)...); }
	
	template <typename ... Args>
	auto find_in_col(Args&& ... args) -> std::optional< pair< size_t, F > >{ 
		return m.find_in_col(std::forward<Args>(args)...); 
	}
		
	template <typename ... Args>
	void add_cols(Args&& ... args){ m.add_cols(std::forward<Args>(args)...); }
	
	// unneeded
	template <typename ... Args>
	void scale_col(Args&& ... args){ }
	
	void add_scaled_col(size_t i, size_t j, size_t k, F lambda = true){
		if (i != k && j != k){ throw std::invalid_argument("i or j must equal k."); }
		add_cols(i,j,k);
	}
	
	template <typename ... Args>
	F operator()(Args&& ... args){ return m(std::forward<Args>(args)...); }
	
	template <typename ... Args>
	void assign_column(Args&& ... args){
		m.assign_column(std::forward<Args>(args)...);
	}
	
};




PYBIND11_MODULE(SparseMatrix, m) {
  m.doc() = "pybind11 sparse matrix plugin";

  // Sparse matrix class
  py::class_<PspFloatMatrix>(m, "PspFloatMatrix")
   .def(py::init<const size_t, const size_t>())
   .def_readonly("cto", &PspFloatMatrix::cto)
   .def_readonly("otc", &PspFloatMatrix::otc)
   .def_readonly("size", &PspFloatMatrix::size)
	 .def_readonly("nnz", &PspFloatMatrix::nnz)
   .def("add_cols", &PspFloatMatrix::add_cols)
   .def("permute_rows", static_cast<void (PspFloatMatrix::*)(const vector< size_t >)>(&PspFloatMatrix::permute_rows), "permute the rows")
   .def("permute_cols", static_cast<void (PspFloatMatrix::*)(const vector< size_t >)>(&PspFloatMatrix::permute_cols), "permute the cols")
  //  .def("print", [](const PspFloatMatrix& m){
  //     py::scoped_ostream_redirect stream(
  //       std::cout,                                // std::ostream&
  //       py::module_::import("sys").attr("stdout") // Python output
  //     );
  //     m.print(std::cout);
  //  })

  //  .def([](py::init([](vector< float > data, vector< size_t > row_ind, vector< size_t > col_ind) {
  //     return std::unique_ptr<Example>(new Example(arg));
  //   })))
  ;
  m.def("print_matrix", [](PspFloatMatrix* m) -> void {
    py::scoped_ostream_redirect stream(
      std::cout,                                // std::ostream&
      py::module_::import("sys").attr("stdout") // Python output
    );
    m->print(std::cout);
  });

  // Reduction algorithm
  m.def("phcol", [](PspFloatMatrix* R1, PspFloatMatrix* V1, PspFloatMatrix* R2, PspFloatMatrix* V2, 
    vector< size_t > I1, vector< size_t > I2
  ) -> void {
    auto r1 = FloatMatrix(*R1);
    auto v1 = FloatMatrix(*V1);
    auto r2 = FloatMatrix(*R2);
    auto v2 = FloatMatrix(*V2);
    pHcol_local(r1,v1,r2,v2, I1.begin(), I1.end(), I2.begin(), I2.end(), []() -> bool {
      return false; 
    });
  });

  // Vineyards
  // m.def()

};

/*
<%
cfg['extra_compile_args'] = ['-std=c++20']
setup_pybind11(cfg)
%>
*/
