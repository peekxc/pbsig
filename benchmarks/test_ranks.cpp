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

void read_dataset(vector< uint64_t >& er, vector< uint64_t >& tr){
  std::ifstream edgefile("../data/edge_ranks_colex_10.txt");
  uint64_t edge_rank; 
  while (edgefile >> edge_rank){
    er.push_back(edge_rank);
  }
  std::ifstream trianglefile("../data/triangle_ranks_colex_10.txt");
  uint64_t triangle_rank; 
  while (trianglefile >> triangle_rank){
    tr.push_back(triangle_rank);
  }
}


void test_rank_it1(vector< uint64_t >& er, vector< uint64_t >& tr){
  // auto R = RankRange< 2 >(500, tr);
  // R.ranks = tr;
  // R.n = 500; 
  // for (auto r: R){
  //   std::cout << r[0] << ", " << r[1] << ", " << r[2] << std::endl;
  // }
}


auto test_ranking_lex() -> std::vector< size_t > {
  std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 };
  std::vector< size_t > ranks(int(triangles.size()/3)); 
  // auto inserter = std::back_inserter(ranks);
  combinatorial::rank_lex(triangles.begin(), triangles.end(), 10, 3, &ranks[0]);
  return ranks; 
}

auto test_ranking_colex() -> std::vector< size_t > {
  // std::vector< size_t > triangles = { 2,1,0,  6,5,4,  6,5,3 }; // rank_colex([])
  std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 }; // rank_colex([])
  std::vector< size_t > ranks(int(triangles.size()/3)); 
  // combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);
  combinatorial::rank_colex< false > (triangles.begin(), triangles.end(), 3, &ranks[0]);
  return ranks; 
}

void test_unranking_colex() {
  std::vector< size_t > triangles = { 2,1,0,  6,5,4,  6,5,3 }; // rank_colex([])
  // std::vector< size_t > triangles = { 0,1,2,  4,5,6,  3,5,6 }; // rank_colex([])
  std::vector< size_t > ranks(int(triangles.size()/3)); 
  // combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);
  
  // Rank them 
  combinatorial::rank_colex< true > (triangles.begin(), triangles.end(), 3, &ranks[0]);
  for (auto r: ranks){ std::cout << r << ", "; }; std::cout << std::endl; // should be in: [0, 100, 90]

  // Unrank them 
  auto max_n = *std::max_element(triangles.begin(), triangles.end());
  std::vector< size_t > new_triangles(ranks.size()*3);
  combinatorial::unrank_colex(ranks.begin(), ranks.end(), 3, max_n+1, &new_triangles[0]);

  for (size_t i = 0; i < new_triangles.size(); i += 3){
    std::cout << new_triangles[i] << ", " << new_triangles[i+1] << ", " << new_triangles[i+2] << std::endl;
  }
}

void colex_unranking(){
  const index_t n = 10; 
  const index_t dim = 2; 
  const auto bc_tbl = binomial_coeff_table(n,dim+1);

  // Read in data set 
  auto er = std::vector<uint64_t>();
  auto tr = std::vector<uint64_t>();
  read_dataset(er, tr);
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;

  std::cout << "Triangles: " << std::endl;
  auto R = RankRange< 2 >(n, tr);
  for (auto r: R){
    std::cout << r[0] << ", " << r[1] << ", " << r[2] << std::endl;
  }

  std::cout << "Triangles: " << std::endl;
  std::array< index_t, 3 > t_labels; 
  for (index_t i = 0; i < tr.size(); ++i){
    get_simplex_vertices(tr[i], 2, n, &t_labels[0], bc_tbl);
    std::cout << i << ": " << t_labels[0] << ", " << t_labels[1] << ", " << t_labels[2] << std::endl;
  }

  // [(0, 1),(0, 6),(1, 2),(1, 6),(3, 4),(3, 5),(3, 7),(3, 8),(3, 9),(4, 5),(4, 7),(4, 8),(5, 7),(5, 9),(7, 8),(7, 9)]
  std::cout << "Edges: " << std::endl;
  std::array< index_t, 2 > e_labels; 
  for (index_t i = 0; i < er.size(); ++i){
    get_simplex_vertices(er[i], 1, n, &e_labels[0], bc_tbl);
    std::cout << i << ": " << e_labels[0] << ", " << e_labels[1] << std::endl;
  }
}


int main(){
  auto t_ranks = test_ranking_lex();
  for (auto r: t_ranks){ std::cout << r << ", "; }; std::cout << std::endl; // should be in: [0, 100, 90]
  t_ranks = test_ranking_colex();
  for (auto r: t_ranks){ std::cout << r << ", "; }; std::cout << std::endl; // should be in: [0, 100, 90]
  
  test_unranking_colex();
  return 0; 
}