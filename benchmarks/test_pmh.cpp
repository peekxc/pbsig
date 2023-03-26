#include <concepts> 
#include "pmh.h"
#include "hash_table5.hpp"
#include "splex_ranges.h"
#include <iostream> 
#include <type_traits>
#include <fstream>

void read_dataset(vector< uint64_t >& er, vector< uint64_t >& tr, std::string e_ranks_fn, std::string t_ranks_fn){
  std::ifstream edgefile(e_ranks_fn);
  uint64_t edge_rank; 
  while (edgefile >> edge_rank){
    er.push_back(edge_rank);
  }
  std::ifstream trianglefile(t_ranks_fn);
  uint64_t triangle_rank; 
  while (trianglefile >> triangle_rank){
    tr.push_back(triangle_rank);
  }
}

template< IntegralHashTable H >
void do_something(H h){
  std::cout << h[0] << std::endl;
  return; 
}
// typename T = typename std::invoke_result< H, int >::type >

// template< typename H >  
// requires IntegralHashTable< H, int > 
// struct MyStruct {
//   H table; 
//   MyStruct(H tbl) : table(tbl) {}
//   size_t operator()(int k){
//     return table(k);
//   }
// };


int main(){
  auto er = std::vector< uint64_t >();
  auto tr = std::vector< uint64_t >();
  read_dataset(er, tr, "../data/edge_ranks_lex_500.txt", "../data/triangle_ranks_lex_500.txt");
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;
  
  auto key_values = vector< std::pair< uint64_t, size_t > >();
  key_values.reserve(tr.size());
  size_t i = 0; 
  for (auto r: er){ key_values.push_back(std::make_pair(r, i++)); }

  std::unordered_map< uint64_t, size_t > ht1(key_values.begin(), key_values.end());
  emhash5::HashMap< uint64_t, size_t > ht2(key_values.begin(), key_values.end());

  auto t_rng = RankRange< 2, true, uint64_t >(tr, 500);
  float sum = 0; 
  size_t cc = 0; 
  for (auto t_it = t_rng.begin(); t_it != t_rng.end(); ++t_it){
    t_it.boundary< true >([&](uint64_t face_rank){
      // sum += std::pow(-1, cc++) * ht1[face_rank];
      sum += std::pow(-1, cc++) * ht2[face_rank];
    });
  }
  std::cout << sum << std::endl;

  return 0; 
}