#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "laplacian.h"

// Namespace directives and declarations
namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal
using namespace combinatorial;

// Type aliases + alias templates 
using std::function; 
using std::vector;
using std::array;
using std::unordered_map;
using uint_32 = uint_fast32_t;
using uint_64 = uint_fast64_t;



// Package: pip install --no-deps --no-build-isolation --editable .
// Compile: clang -Wall -fPIC -c src/pbsig/laplacian_grad.cpp -std=c++20 -Iextern/pybind11/include -isystem /Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include -I/Users/mpiekenbrock/opt/miniconda3/envs/pbsig/include/python3.9 -I/Users/mpiekenbrock/pbsig/include -I/Users/mpiekenbrock/pbsig/extern/eigen
// PYBIND11_MODULE(_laplacian_grad, m) {
//   m.doc() = "Laplacian gradient module";
//   m.grad() = "_rmatmat", [](const Class& L, const array_t_FF& X){ return _matmat(L, X); }
// }