#include "splex_ranges.h"
#include <iostream>


void test_rank_it1(vector< uint64_t >& er, vector< uint64_t >& tr){
  auto R = RankRange< 2 >(500, tr);
  // R.ranks = tr;
  // R.n = 500; 
  for (auto r: R){
    std::cout << r[0] << ", " << r[1] << ", " << r[2] << std::endl;
  }
}

void test(){

  const index_t n = 10; 
  const index_t dim = 2; 
  const auto bc_tbl = binomial_coeff_table(n,dim+1);
  index_t n_edges = bc_tbl(n,2);
  index_t n_triangles = bc_tbl(n,3);
  std::cout << "E: " << n_edges << ", T: " << n_triangles << std::endl;

  std::array< index_t, 3 > t_labels; 
  for (index_t i = 0; i < n_triangles; ++i){
    get_simplex_vertices(i, dim, n, &t_labels[0], bc_tbl);
    std::cout << i << ": " << t_labels[0] << ", " << t_labels[1] << ", " << t_labels[2] << std::endl;
    if (i == 5){
      break;
    }
  }
  
  std::cout << "Edges: " << std::endl;
  std::array< index_t, 2 > e_labels; 
  for (index_t i = 0; i < n_edges; ++i){
    get_simplex_vertices(i, dim-1, n, &e_labels[0], bc_tbl);
    std::cout << i << ": " << e_labels[0] << ", " << e_labels[1] << std::endl; 
  }

  auto s = simplex_boundary_enumerator(0, dim, n, bc_tbl);
  s.set_simplex(0, dim, n);
  while (s.has_next()){
    std::cout << "face index:" << s.next() << std::endl;
  }
}


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


void run_benchmark(vector< uint64_t >& tr){
  const index_t n = 500; 
  const index_t dim = 2; 
  const auto bc_tbl = binomial_coeff_table(n,dim+1);
  index_t n_edges = bc_tbl(n,2);
  index_t n_triangles = bc_tbl(n,3);
  std::cout << "E: " << n_edges << ", T: " << n_triangles << std::endl;

  float s = 0;
  std::array< index_t, 3 > t_labels; 
  for (index_t i = 0; i < tr.size(); ++i){
    get_simplex_vertices(i, dim, n, &t_labels[0], bc_tbl);
    s += std::pow(-1, i % 2)*(t_labels[0]+t_labels[1]+t_labels[2]);
  }
  std::cout << s << std::endl;
}

// get_max_vertex
// get_simplex_vertices(idx,  dim, n,  out) 
// simplex_boundary_enumerator
int main() {
  vector< uint64_t > er, tr; 
  read_dataset(er, tr);
  std::cout << "num edges: " << er.size() << ", " << "num triangles: " << tr.size() << std::endl;

  //auto L = UpLaplacian< 1, float >(tr, er, 500); // construct  
  for (int i = 0; i < 15; ++i){
    run_benchmark(tr);
  }

  return 0;
}