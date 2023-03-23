#include "combinatorial.h"
#include <iostream>
#include <random>
#include <vector> 
#include <cassert>

void benchmark_bc_1(const size_t k, const size_t repeats=10000){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 500);
  for (size_t r = 0; r < repeats; ++r) {
    size_t c = 0, cc = 0;
    for (size_t i = 0; i < 5000; ++i){
      c += std::pow(-1, cc++) * combinatorial::BinomialCoefficient(k+dis(gen), k);
    }
  }
}

void benchmark_bc_2(const size_t k, const size_t repeats=10000){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 500);
  for (size_t r = 0; r < repeats; ++r) {
    size_t c = 0, cc = 0;
    for (size_t i = 0; i < 5000; ++i){
      c += std::pow(-1, cc++) * combinatorial::binom(k+dis(gen), k);
    }
  }
}

typedef int64_t index_t;
struct binomial_coeff_table {
  std::vector<std::vector<index_t>> B;
  binomial_coeff_table(index_t n, index_t k) : B(k + 1, std::vector<index_t>(n + 1, 0)) {
    for (index_t i = 0; i <= n; ++i) {
      B[0][i] = 1;
      for (index_t j = 1; j < std::min(i, k + 1); ++j)
        B[j][i] = B[j - 1][i - 1] + B[j][i - 1];
      if (i <= k) B[i][i] = 1;
      //check_overflow(B[std::min(i >> 1, k)][i]);
    }
  }

  index_t operator()(index_t n, index_t k) const {
    if (!(k < B.size() && n < B[k].size()  && n >= k - 1)){
      std::cout << "n: " << n << ", k: " << k << ", B size: " << B.size() << ", B[k].size: " << B[k].size() << std::endl;
    }
    assert(k < B.size() && n < B[k].size()  && n >= k - 1);
    return B[k][n];
  }
};

void benchmark_bc_3(const size_t k, const size_t repeats=10000){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 500);
  auto bc = binomial_coeff_table(500, k);
  for (size_t r = 0; r < repeats; ++r) {
    size_t c = 0, cc = 0;
    for (size_t i = 0; i < 5000; ++i){
      const size_t n = dis(gen); 
      if (n > k){
        c += std::pow(-1, cc++) * bc(n, k);  
      }
    }
  }
}

void test_bc(){
  for (size_t n = 0; n < 128; ++n){
    for (size_t k = 0; k < 8; ++k){
      size_t bc1 = combinatorial::binom(n,k);
      // size_t bc2 = combinatorial::binomial_coeff_(n,k);
      size_t bc2 = combinatorial::BinomialCoefficient(n,k);
      if (bc1 != bc2){
        std::cout << bc1 << std::endl; 
        std::cout << bc2 << std::endl; 
        std::cout << "C(" << n << "," << k << ")" << std::endl; 
      }
      assert( bc1 == bc2 );
    }
  }
}
  // const size_t N = 16; 
  // const size_t K = 4;   
  // size_t c = 0, cc = 0;
  // for (size_t k = 0; k < K; ++k){
  //   for (size_t n = 0; n < N; ++n){
  //     auto bc = combinatorial::BinomialCoefficient(n,k);
  //     std::cout << "n: " << n << ", k: " << k << ": " << bc << std::endl;
  //     c += std::pow(-1, cc++) * bc;
  //   }
  // }
  // std::cout << "n: " << 128 << ", k: " << 5 << ": " << combinatorial::BinomialCoefficient(128,5) << std::endl; // 264566400


int main(){
  test_bc();
  // benchmark_bc_1(3);
  // benchmark_bc_2(3);
  // benchmark_bc_3(3);
  return 0; 
}


