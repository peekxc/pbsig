#include <concepts> 
#include <vector>
#include <array> 
#include <cinttypes>  // uint32_t 
#include <random>     // rand
#include <algorithm>  // max 
#include <iostream> 
#include <optional> 
#include <cassert>

using std::pair; 
using std::vector;
using std::array; 

template < typename F, typename T >
concept IntegerHashFunction = std::regular_invocable<F, T> && std::convertible_to< std::invoke_result_t< F, T >, uint32_t >; 

template < typename Table, typename Key = typename Table::key_type >
concept IntegralHashTable = requires(Table h, Key k) {
  { h.operator[](k) } -> std::integral;
};

// Miller-Rabin primality test
bool is_prime(int n){
  if (n < 2) return false;
  if (n == 2) return true;
  if (n % 2 == 0) return false;
  for (int i=3; (i*i)<=n; i+=2){
    if (n % i == 0) return false;
  }
  return true; 
}

// Generate n primes above m
void gen_primes_above(int m, const size_t n, vector< uint64_t >& primes){
  primes.reserve(n);
  while(primes.size() < n){
    // Bertrand's postulate
    for (uint32_t p = m; p < 2*m - 2; ++p){
      if (is_prime(p)){ primes.push_back(p); }
      if (primes.size() >= n){ break; }
    }
    m = primes.back() + 1; 
  }
}


// ## Universal Hash Function (Linear Congruential Generator)
// ## https://en.wikipedia.org/wiki/Linear_congruential_generator
struct LCG {
  mutable uint64_t a;  
  mutable uint64_t b;  
  mutable uint64_t p;   
  mutable uint32_t m;  
  const vector< uint64_t >& primes;

  LCG(const vector< uint64_t >& p) : a(0), b(0), p(0), m(0), primes(p) { }
  
  void randomize(uint32_t _m){
    m = _m; 
    p = primes[rand() % primes.size()];
    a = std::max(uint64_t(1), uint64_t(rand() % p)); 
    b = rand() % p; 
  }

  uint32_t operator()(uint32_t x) const {
    // if (x == 0){
    //   std::cout << "a*x" << a*x << ", a*x + b = " << a*x + b << ", ((a*x + b) \% p) = " << ((a*x + b) % p) << ", ((a*x + b) \% p) \% m = " << ((a*x + b) % p) % m << std::endl;
    //   std::cout << "a: " << a << ", b: " << b << ", p: " << p << ", m: " << m << std::endl;
    // }
    uint64_t r = a*x;
    if (a != 0 && r / a != x) { throw std::overflow_error("Detected overflow in generated LCG. Please try using a smaller set of primes."); }
    return ((r + b) % p) % m;
    
  }
  void print(uint32_t x){
    std::cout << "a: " << a << ", b: " << b << ", p: " << p << ", m: " << m << std::endl;
    std::cout << "x = " << x << ", a*x = " << a*x << ", a*x + b = " << a*x + b << ", ((a*x + b) \% p) = " << ((a*x + b) % p) << ", ((a*x + b) \% p) \% m = " << ((a*x + b) % p) % m << std::endl;
  }
};

template< uint16_t d >
struct LCG_k {
  mutable array< uint64_t, d > a;  
  mutable uint64_t p;   
  mutable uint32_t m;  
  const vector< uint64_t >& primes;

  LCG_k(const vector< uint64_t >& p) : p(0), m(0), primes(p) { }
  
  void randomize(uint32_t _m){
    m = _m; 
    p = primes[rand() % primes.size()];
    for (size_t i = 0; i < d; ++i){
      a[i] = std::max(uint64_t(1), uint64_t(rand() % p)); 
    }
  }

  uint32_t operator()(uint32_t x) const {
    uint32_t w = 1; 
    uint64_t r = 0; 
    for (size_t i = 0; i < d; ++i){
      r += (a[i]*w) % p;
      w *= x;
    }
    return r % m;
  }
};


template< std::integral K >
struct LcgDagHash {
  const uint32_t m;  // table size 
  const array< uint64_t, 2 > a;  
  const array< uint64_t, 2 > b;  
  const array< uint64_t, 2 > p;   
  const vector< uint32_t > g; 
  // LcgDagHash(){};  
  LcgDagHash(const vector< uint32_t >& _g,
    uint64_t _a1, uint64_t _a2, 
    uint64_t _b1, uint64_t _b2, 
    uint64_t _p1, uint64_t _p2,
    uint32_t _m
  ) : g(_g.begin(), _g.end()), a{_a1, _a2}, b{_b1, _b2}, p{_p1, _p2}, m(_m)
  { 
  
  }

  void print(){
    std::cout << "f1 :=  a: " << a[0] << ", b: " << b[0] << ", p: " << p[0] << ", m: " << m << std::endl;
    std::cout << "f2 :=  a: " << a[1] << ", b: " << b[1] << ", p: " << p[1] << ", m: " << m << std::endl;
  }
 
  constexpr uint32_t f1(K x) const noexcept {
    return (((a[0]*x + b[0]) % p[0]) % m);
  }
  constexpr uint32_t f2(K x) const noexcept {
    return (((a[1]*x + b[1]) % p[1]) % m);
  }
  
  [[nodiscard]]
  constexpr uint32_t operator()(K x, bool verbose = false) const noexcept {
    if (!verbose){
      return (g[f1(x)] + g[f2(x)]) % m;
    } else {
      std::cout << "x = " << x << std::endl;
      std::cout << "[f1] (((a[0]*x + b[0]) \% p[0]) \% m): " << f1(x) << std::endl;
      std::cout << "[f2] (((a[1]*x + b[1]) \% p[1]) \% m): " << f2(x) << std::endl;
      std::cout << "g[f1(x)]: " << g[f1(x)] << std::endl; 
      std::cout << "g[f2(x)]: " << g[f2(x)] << std::endl; 
      uint32_t r = g[f1(x)] + g[f2(x)];
      std::cout << "r = " << r << std::endl;
      std::cout << "r \% m = " << r % m << std::endl;
      return r % m;
    }
    
  }
};

// Modulo that works with negative integers. Matches python's % operation, differs from C's bitwise %. 
constexpr uint32_t mod(int32_t n, int32_t m){
  return ((n % m) + m) % m;
}

struct IntSaltHash {
  mutable vector< uint64_t > salt;
  uint64_t m;
  IntSaltHash() : salt{} {}
  void randomize(uint64_t m) {
    this->m = m;
    salt.clear();
  }
  uint64_t operator()(uint64_t key) const {
    uint64_t hash = 0;
    size_t i = 0;
    while (key != 0) {
      uint64_t digit = key % 10;
      key /= 10;
      if (i >= salt.size()) {
        salt.push_back(std::rand() % (m - 1) + 1);
      }
      hash += salt[i] * digit;
      i++;
    } 
    return hash % m;
  }
};

template< std::integral K >
struct PerfectHashDAG {
  using node_t = pair< uint32_t, uint32_t >; 
  vector< vector< node_t > > adj;
  vector< uint32_t > vertex_values; 
  
  PerfectHashDAG(){
    // srand(1234);
    adj = vector< vector< node_t > >();
    // g[((42*x+246)%397)%13]+g[((83*x+137)%353)%13])%13
  }

  void reset(){
    for (size_t i = 0; i < adj.size(); ++i){ adj[i].clear(); }
  }
  void resize(const size_t N){
    adj.resize(N);
  }

  template< typename InputIt, IntegerHashFunction< uint64_t > H1, IntegerHashFunction< uint64_t > H2 > 
  void connect_all(InputIt k_it, const InputIt k_end, H1 f1, H2 f2){
    const uint32_t N = std::distance(k_it, k_end);
    for (uint32_t i = 0; k_it != k_end; ++k_it, ++i){
      uint32_t v0 = f1(*k_it);
      uint32_t v1 = f2(*k_it);
      // std::cout << "f1(" << *k_it << ") -> v0 = " << v0 << std::endl;
      // std::cout << "f2(" << *k_it << ") -> v1 = " << v1 << std::endl;
      adj.at(v0).push_back(std::make_pair(v1, i));
      adj.at(v1).push_back(std::make_pair(v0, i));
    }
  }

  void print_adj(){
    for (size_t a = 0; a < adj.size(); ++a){
      std::cout << a << ": ";
      for (auto neighbor: adj[a]){
        std::cout << neighbor.first << ", ";
      }
      std::cout << std::endl;
    }
  }

  bool assign_values(const size_t N){
    vertex_values.resize(N);
    std::fill(vertex_values.begin(), vertex_values.end(), 0);
    auto visited = vector< bool >(N, false);
    
    // Loop over all vertices, taking unvisited ones as roots
    for (size_t root = 0; root < N; ++root){
      if (visited[root]){ continue; }
      vector< pair< int32_t, uint32_t > > to_visit = { std::make_pair(-1, root) };
      while (to_visit.size() > 0){
        auto [parent, vertex] = to_visit.back();
        visited.at(vertex) = true;
        to_visit.pop_back();
        // std::cout << "dfs-visiting..." << parent << "->" << vertex << std::endl;

        // DFS: Loop over adjacent vertices, but skip the vertex we arrived here from the first time it is encountered.
        bool skip = true;
        for (auto p: adj.at(vertex)){
          auto [neighbor, edge_value] = p;
          // std::cout << "searching neighbors of " << vertex << "..." << neighbor << "[" << edge_value << "]" <<  std::endl;
          if (skip && neighbor == parent){
            skip = false; 
            continue; 
          }
          if (visited.at(neighbor)){ return false; } // graph is cyclic! 
          // std::cout << "pushing edge: " << vertex << ", " << neighbor << std::endl;
          to_visit.push_back(std::make_pair(vertex, neighbor));
          // std::cout << "Assigning: g[" << neighbor << "] = (" << edge_value << "- g[" << vertex << "]) % " << N << " == " << uint32_t((int32_t(edge_value) - int32_t(vertex_values.at(vertex))) % N) << std::endl;
          // vertex_values.at(neighbor) = uint32_t((int32_t(edge_value) - int32_t(vertex_values.at(vertex))) % N);  // Assignment step
          vertex_values.at(neighbor) = static_cast< uint32_t >(mod(int32_t(edge_value) - int32_t(vertex_values.at(vertex)), N));
        }
      }
    }
    return true; // success
  }



  template< typename InputIt, IntegerHashFunction< uint64_t > H1,  IntegerHashFunction< uint64_t > H2 >
  // requires IntegerHashFunction< H1, uint64_t >
  auto build_hash(InputIt k_it, const InputIt k_end, float mult_max, size_t n_tries, H1 f1, H2 f2, bool verbose = true) -> bool {
    const uint32_t N = std::distance(k_it, k_end);
    if (N == 0){ return false;  }
    size_t n_attempts = 0; 
    bool success = false;
    float step_sz = (ceil(mult_max*N)-N)/n_tries;
    for (size_t i = 0; i < n_tries; ++i){
      uint32_t NG = uint32_t(N + i*step_sz);
      
      f1.randomize(NG);
      f2.randomize(NG);
      reset();
      resize(NG);
      connect_all(k_it, k_end, f1, f2);
      if ((success = assign_values(NG))){
        if (verbose){ std::cout << "Success @ iteration " << i << " w/ size " << NG << std::flush << std::endl; }
        break; 
      }
    }
    return success; 
  }
};
