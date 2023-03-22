

#include "splex_ranges.h"
#include "laplacian.h"

#include <iostream>
#include <fstream>
#include <cinttypes>
#include <random>
#include <algorithm>
#include <iterator>

using std::uint64_t;
using std::vector;

void read_dataset(vector< uint64_t >& er, vector< uint64_t >& tr){
  std::ifstream edgefile("../data/edge_ranks.txt");
  uint64_t edge_rank; 
  while (edgefile >> edge_rank){
    er.push_back(edge_rank);
  }
  std::ifstream trianglefile("../data/triangle_ranks.txt");
  uint64_t triangle_rank; 
  while (trianglefile >> triangle_rank){
    tr.push_back(triangle_rank);
  }
}

// static may provoke some thread-safety issues.
static void generate_data(std::vector< float >& data, size_t size) {
  using value_type = float; // data::value_type;
  static std::uniform_real_distribution<value_type> distribution(
      -10.0, // std::numeric_limits<value_type>::min(),
       10.0  // std::numeric_limits<value_type>::max()
  );
  static std::default_random_engine generator;
  std::generate(data.begin(), data.end(), []() { return distribution(generator); });
}

template< typename Laplacian >
float run_benchmark(Laplacian& L, size_t n){
  float y_sum = 0.0; 
  auto x = vector< float >(L.np, 1.0);
  for (size_t i = 0; i < n; ++i){
    generate_data(x, L.np);
    L.__matvec(x.data());
    y_sum += std::pow(-1.0, i) * std::accumulate(L.y.begin(), L.y.end(), 0.0);
  }
  return y_sum;
}



int main(){
  vector< uint64_t > er, tr; 
  read_dataset(er, tr);
  std::cout << "num edges: " << er.size() << ", " << "num triangles: " << tr.size() << std::endl;

  auto L = UpLaplacian< 1, float >(tr, er, 500); // construct  
  run_benchmark(L, 100);

  std::cout << "Size of output vec: " << L.y.size() << std::endl; 
  std::cout << "First elements after matvec(1): ";
  auto x = vector< float >(L.np, 1.0);
  L.__matvec(x.data());
  for (size_t i = 0; i < 35; i++){
    std::cout << L.y[i] << ", ";
  }
  std::cout << std::endl;

  

  return 0; 
}
