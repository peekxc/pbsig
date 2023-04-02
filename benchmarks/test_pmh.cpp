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

void read_g(vector< uint32_t >& g){
  std::ifstream g_file("../data/er_g_colex_500.txt");
  uint32_t i; 
  while (g_file >> i){ g.push_back(i); }
}


template< std::integral K > 
void test_pmh(vector< K > keys){
  auto pmh = PerfectHashDAG< K >();
  std::cout << "building hash..." << std::endl;
  auto h_opt = pmh.build_hash(keys.begin(), keys.end(), 3.5, 10000, 1000);
  if (h_opt.has_value()){
    auto h = *h_opt; 
    size_t c = 0; 
    for (auto k: keys){
      std::cout << k << " -> " << h(k) << std::endl;
      c++;
      if (c > 15){ break; }
    }
  }
}


int main(){
  // std::vector< uint64_t > r = { 0, 5, 13, 45, 82, 102, 103, 114, 115, 156 };
  // test_pmh(r);

  auto er = std::vector< uint64_t >();
  auto tr = std::vector< uint64_t >();
  read_dataset(er, tr, "../data/edge_ranks_colex_500.txt", "../data/triangle_ranks_colex_500.txt");
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;
  test_pmh< uint64_t >(er);
  // auto t_rng = RankRange< 2, true, uint64_t >(tr, 500);

  // auto key_values = vector< std::pair< uint64_t, size_t > >();
  // key_values.reserve(tr.size());
  // size_t i = 0; 
  // for (auto r: er){ key_values.push_back(std::make_pair(r, i++)); }

  // std::unordered_map< uint64_t, uint32_t > ht1(key_values.begin(), key_values.end());
  // emhash5::HashMap< uint64_t, uint32_t > ht2(key_values.begin(), key_values.end());

  // // LCG DAG 
  // std::vector< uint32_t > g; 
  // read_g(g);
  // std::cout << "g size: " << g.size() << std::endl;
  // // std::vector< int > mul = { 5618, 3170 }; 
  // // std::vector< int > add = { 9597, 7593 }; 
  // // std::vector< int > mod = { 11699, 11699 }; 
  // // LCG_PMF< uint64_t, uint64_t, uint32_t >(
  // //   6139, 2, g, vector< int > M, vector< int > A, vector< int > U
  // // );
  // // (g[((5618*x+9597)%11699)%6139]+g[((3170*x+7593)%8627)%6139])%6139

  // // Iterate through boundaries of rank range
  // float sum = 0; 
  // const std::array< int, 2 > sign = { 1, -1 }; 

  // for (size_t i = 0; i < 10000; ++i){
  //   uint32_t cc = 0; 
  //   for (auto t_it = t_rng.begin(); t_it != t_rng.end(); ++t_it, ++cc){
  //     t_it.boundary< true >([&](uint64_t face_rank){
  //       // auto hv = ht1[face_rank];
  //       // auto hv = ht2[face_rank];
  //       auto hv = uint32_t(g[((5618*face_rank+9597)%11699)%6139]+g[((3170*face_rank+7593)%8627)%6139]) % 6139u;
  //       sum += sign[cc++ % 2] * hv;
  //     });
  //   }
  // }
  
  // std::cout << sum << std::endl;

  return 0; 
}

// if (hv != ht2[face_rank]){
//   std::cout << "r: " << face_rank << std::endl;
//   std::cout << "rhf: " << ht2[face_rank] << std::endl;
//   std::cout << "pmh: " << hv << std::endl;
// }
// if (cc < 15){ std::cout << cc << ", " << sign[cc % 2] << "; "; }