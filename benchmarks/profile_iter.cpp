#include "splex_ranges.h"
#include <iostream>

void read_dataset_labels(vector< uint16_t >& labels, std::string labels_fn){
  std::ifstream label_file(labels_fn);
  uint16_t label; 
  while (label_file >> label){
    labels.push_back(label);
  }
}

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

void benchmark_lex_unranking(){
  auto er = std::vector<uint64_t>();
  auto tr = std::vector<uint64_t>();
  read_dataset(er, tr, "../data/edge_ranks_lex_500.txt", "../data/triangle_ranks_lex_500.txt");
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;

  for (size_t i = 0; i < 100; ++i){
    float c = 0;
    size_t cc = 0;
    auto t_rng = RankRange< 2, false, uint64_t >(500, tr);
    for (auto t: t_rng){
      // std::cout << t[0] << ", " << t[1] << ", " << t[2] << std::endl;
      c += std::pow(-1, cc++)*(t[0]+t[1]+t[2]);
    }
  }
}

void benchmark_colex_unranking(){
  auto er = std::vector<uint64_t>();
  auto tr = std::vector<uint64_t>();
  read_dataset(er, tr, "../data/edge_ranks_colex_500.txt", "../data/triangle_ranks_colex_500.txt");
  std::cout << "E: " << er.size() << ", T: " << tr.size() << std::endl;

  auto t_rng = RankRange< 2, true, uint64_t >(500, tr);
  for (size_t i = 0; i < 100; ++i){
    float c = 0;
    size_t cc = 0;
    for (auto t: t_rng){
      // std::cout << t[0] << ", " << t[1] << ", " << t[2] << std::endl;
      c += std::pow(-1, cc++)*(t[0]+t[1]+t[2]);
    }
  }
}

void benchmark_simplex_range(vector< uint16_t >& triangles, const size_t n){
  auto s_rng = SimplexRange< 2 >(triangles, n);
  float sum = 0; 
  for (size_t i = 0; i < 100; ++i){
    size_t cc = 0; 
    sum = 0;
    for (auto s_it = s_rng.begin(); s_it != s_rng.end(); ++s_it){
      s_it.boundary< true >([&](auto face_rank){ sum += std::pow(-1, cc++)*face_rank; });
    }
  }
  std::cout << sum << std::endl;
}

// void test(){

//   const index_t n = 10; 
//   const index_t dim = 2; 
//   const auto bc_tbl = binomial_coeff_table(n,dim+1);
//   index_t n_edges = bc_tbl(n,2);
//   index_t n_triangles = bc_tbl(n,3);
//   std::cout << "E: " << n_edges << ", T: " << n_triangles << std::endl;

//   std::array< index_t, 3 > t_labels; 
//   for (index_t i = 0; i < n_triangles; ++i){
//     get_simplex_vertices(i, dim, n, &t_labels[0], bc_tbl);
//     std::cout << i << ": " << t_labels[0] << ", " << t_labels[1] << ", " << t_labels[2] << std::endl;
//     if (i == 5){
//       break;
//     }
//   }
  
//   std::cout << "Edges: " << std::endl;
//   std::array< index_t, 2 > e_labels; 
//   for (index_t i = 0; i < n_edges; ++i){
//     get_simplex_vertices(i, dim-1, n, &e_labels[0], bc_tbl);
//     std::cout << i << ": " << e_labels[0] << ", " << e_labels[1] << std::endl; 
//   }

//   auto s = simplex_boundary_enumerator(0, dim, n, bc_tbl);
//   s.set_simplex(0, dim, n);
//   while (s.has_next()){
//     std::cout << "face index:" << s.next() << std::endl;
//   }
// }

// void run_benchmark(vector< uint64_t >& tr){
//   const index_t n = 500; 
//   const index_t dim = 2; 
//   const auto bc_tbl = binomial_coeff_table(n,dim+1);
//   index_t n_edges = bc_tbl(n,2);
//   index_t n_triangles = bc_tbl(n,3);
//   std::cout << "E: " << n_edges << ", T: " << n_triangles << std::endl;

//   float s = 0;
//   std::array< index_t, 3 > t_labels; 
//   for (index_t i = 0; i < tr.size(); ++i){
//     get_simplex_vertices(i, dim, n, &t_labels[0], bc_tbl);
//     s += std::pow(-1, i % 2)*(t_labels[0]+t_labels[1]+t_labels[2]);
//   }
//   std::cout << s << std::endl;
// }

// get_max_vertex
// get_simplex_vertices(idx,  dim, n,  out) 
// simplex_boundary_enumerator
int main() {
  // benchmark_lex_unranking();
  // benchmark_colex_unranking();

  vector< uint16_t > triangles = std::vector< uint16_t >();
  read_dataset_labels(triangles, "/Users/mpiekenbrock/pbsig/data/triangle_labels_500.txt");
  benchmark_simplex_range(triangles, 500);

  // auto s_rng = SimplexRange< 2 >(triangles, 500);
  // size_t i = 0; 
  // for (auto s: s_rng){
  //   std::cout << s[0] << "," << s[1] << "," << s[2] << std::endl;
  //   i++;
  //   if (i == 10){
  //     break;
  //   }
  // }

  // vector< uint64_t > er, tr; 
  // read_dataset(er, tr);
  // std::cout << "num edges: " << er.size() << ", " << "num triangles: " << tr.size() << std::endl;

  // //auto L = UpLaplacian< 1, float >(tr, er, 500); // construct  
  // for (int i = 0; i < 15; ++i){
  //   run_benchmark(tr);
  // }

  return 0;
}