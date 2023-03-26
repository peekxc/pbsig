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

template< bool colex = true >
void test_unranking(){
  auto er = std::vector<uint64_t>();
  auto tr = std::vector<uint64_t>();
  if (colex){
    read_dataset(er, tr, "../data/edge_ranks_colex_10.txt", "../data/triangle_ranks_colex_10.txt");
  } else {
    read_dataset(er, tr, "../data/edge_ranks_lex_10.txt", "../data/triangle_ranks_lex_10.txt");
  }
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;

  std::cout << "Edge Rank range: " << std::endl;
  auto e_rng = RankRange< 1, colex, uint64_t >(10, er);
  for (auto e = e_rng.begin(); e != e_rng.end(); ++e){
    std::cout << (*e)[0] << ", " << (*e)[1];
    std::cout << "  (BR): ";
    // See: https://stackoverflow.com/questions/34696351/template-dependent-typename
    // e.template boundary< true >([](auto face_rank){ std::cout << face_rank << ","; });
    std::cout << std::endl;
  }

  std::cout << "Triangle Rank range: " << std::endl;
  auto t_rng = RankRange< 2, colex, uint64_t >(10, tr);
  for (auto t = t_rng.begin(); t != t_rng.end(); ++t) {
    std::cout << (*t)[0] << ", " << (*t)[1] << ", " << (*t)[2]; 
    std::cout << "  (BR): ";
    t.template boundary< true >([](auto face_rank){ std::cout << face_rank << ","; });
    std::cout << std::endl;
  }
}

void test_simplex_range(){
  vector< uint16_t > E_lex = { 0,1,0,3,0,5,0,7,1,3,1,5,1,6,1,7,1,9,2,8,3,5,3,7,4,6,4,9,5,6,5,7,6,7,6,9 };
  vector< uint16_t > T_lex = { 0,1,3,0,1,5,0,1,7,0,3,5,0,3,7,0,5,7,1,3,5,1,3,7,1,5,6,1,5,7,1,6,7,1,6,9,3,5,7,4,6,9,5,6,7 };
  
  auto SE = SimplexRange< 1, false >(E_lex, 10);
  std::cout << "Edges: " << std::endl;
  for (auto s_it = SE.begin(); s_it != SE.end(); ++s_it){
    std::cout << (*s_it)[0] << "," << (*s_it)[1]; 
    std::cout << "  (BR): ";
    s_it.boundary< true >([](auto face_rank){ std::cout << face_rank << ","; });
    std::cout << std::endl;
  }
  
  auto ST = SimplexRange< 2, false >(T_lex, 10);
  std::cout << "Triangles: " << std::endl;
  for (auto s_it = ST.begin(); s_it != ST.end(); ++s_it){
    std::cout << (*s_it)[0] << "," << (*s_it)[1] << "," << (*s_it)[2]; 
    std::cout << "  (BR): ";
    s_it.boundary< true >([](auto face_rank){ std::cout << face_rank << ","; });
    std::cout << std::endl;
  }
}



	//return std::ceil(m * exp(log(r)/m + log(2*pi*m)/2*m + 1/(12*m*m) - 1/(360*pow(m,4)) - 1) + (m-1)/2);

int main(){
  // test_colex_unranking();
  test_unranking< true >();
  test_simplex_range();
  // benchmark_lex_unranking();
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