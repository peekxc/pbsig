#include "splex_ranges.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <cinttypes>
#include <random>
#include <algorithm>
#include <iterator>

using std::uint64_t;
using std::vector;

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

auto test_ranking_lex() -> std::vector< size_t > {
  std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 };
  std::vector< size_t > ranks(int(triangles.size()/3)); 
  combinatorial::rank_lex(triangles.begin(), triangles.end(), 10, 3, &ranks[0]);
  return ranks; 
}

void test_colex_unranking(){
  auto er = std::vector<uint64_t>();
  auto tr = std::vector<uint64_t>();
  read_dataset(er, tr, "../data/edge_ranks_colex_10.txt", "../data/triangle_ranks_colex_10.txt");
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;

  std::cout << "Edge Rank range: " << std::endl;
  auto e_rng = RankRange< 1, true, uint64_t >(10, er);
  for (auto e: e_rng){
    std::cout << e[0] << ", " << e[1] << std::endl;
  }

  std::cout << "Triangle Rank range: " << std::endl;
  auto t_rng = RankRange< 2, true, uint64_t >(10, tr);
  for (auto t: t_rng){
    std::cout << t[0] << ", " << t[1] << ", " << t[2] << std::endl;
  }
}

void test_lex_unranking(){
  auto er = std::vector<uint64_t>();
  auto tr = std::vector<uint64_t>();
  read_dataset(er, tr, "../data/edge_ranks_lex_10.txt", "../data/triangle_ranks_lex_10.txt");
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;

  std::cout << "Edge Rank range: " << std::endl;
  auto e_rng = RankRange< 1, false, uint64_t >(10, er);
  for (auto e: e_rng){
    std::cout << e[0] << ", " << e[1] << std::endl;
  }

  std::cout << "Triangle Rank range: " << std::endl;
  auto t_rng = RankRange< 2, false, uint64_t >(10, tr);
  for (auto t: t_rng){
    std::cout << t[0] << ", " << t[1] << ", " << t[2] << std::endl;
  }
}

int main(){
  // test_colex_unranking();
  // test_lex_unranking();
  benchmark_lex_unranking();
  // auto t_ranks = test_ranking_lex();
  // for (auto r: t_ranks){ std::cout << r << ", "; }; std::cout << std::endl; // should be in: [0, 100, 90]
  // // t_ranks = test_ranking_colex();
  // // for (auto r: t_ranks){ std::cout << r << ", "; }; std::cout << std::endl; // should be in: [0, 100, 90]
  
  // test_unranking_colex();

  // // std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 }; // rank_colex([])
  // // std::vector< size_t > ranks(int(triangles.size()/3)); 
  // // combinatorial::rank_lex(triangles.begin(), triangles.end(), 7, 3, &ranks[0]);

  // std::vector< size_t > triangles = { 2,1,0,  6,5,4,  6,5,3 };
  // std::vector< size_t > ranks(int(triangles.size()/3)); 
  // combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);

  // std::cout << "Ranks: " << std::endl;
  // for (auto r: ranks){ std::cout << r << ", "; }; std::cout << std::endl; 

  // std::cout << "Rank range: " << std::endl;
  // auto rng = RankRange< 2, true, size_t >(7, ranks);
  // for (auto t: rng){
  //   std::cout << t[0] << ", " << t[1] << ", " << t[2] << std::endl;
  // }



  return 0; 
}



// auto test_ranking_colex() -> std::vector< size_t > {
//   // std::vector< size_t > triangles = { 2,1,0,  6,5,4,  6,5,3 }; // rank_colex([])
//   std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 }; // rank_colex([])
//   std::vector< size_t > ranks(int(triangles.size()/3)); 
//   // combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);
//   combinatorial::rank_colex< false > (triangles.begin(), triangles.end(), 3, &ranks[0]);
//   return ranks; 
// }

// void test_unranking_colex() {
//   std::vector< size_t > triangles = { 2,1,0,  6,5,4,  6,5,3 }; // rank_colex([])
//   // std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 }; // rank_colex([])
//   std::vector< size_t > ranks(int(triangles.size()/3)); 
//   // combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);
  
//   // Rank them 
//   combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);
//   for (auto r: ranks){ std::cout << r << ", "; }; std::cout << std::endl; // should be in: [0, 100, 90]

//   // Unrank them 
//   auto max_n = *std::max_element(triangles.begin(), triangles.end());
//   std::vector< size_t > new_triangles(ranks.size()*3);
//   combinatorial::unrank_colex(ranks.begin(), ranks.end(), max_n+1, 3, &new_triangles[0]);

//   for (size_t i = 0; i < new_triangles.size(); i += 3){
//     std::cout << new_triangles[i] << ", " << new_triangles[i+1] << ", " << new_triangles[i+2] << std::endl;
//   }
// }