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
void gen_primes_above(int m, const size_t n, vector< uint32_t >& primes){
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
  mutable uint32_t a;  
  mutable uint32_t b;  
  mutable uint32_t p;   
  mutable uint32_t m;  
  // LCG() : a(0), b(0), p(0), m(0){ }
  
  void randomize(uint32_t _m, const vector< uint32_t >& primes){
    m = _m; 
    p = primes[rand() % primes.size()];
    a = std::max(uint32_t(1), uint32_t(rand() % p)); 
    b = rand() % p; 
  }

  constexpr uint32_t operator()(uint32_t x) const {
    // if (x == 0){
    //   std::cout << "a*x" << a*x << ", a*x + b = " << a*x + b << ", ((a*x + b) \% p) = " << ((a*x + b) % p) << ", ((a*x + b) \% p) \% m = " << ((a*x + b) % p) % m << std::endl;
    //   std::cout << "a: " << a << ", b: " << b << ", p: " << p << ", m: " << m << std::endl;
    // }
    return ((a*x + b) % p) % m;
  }
  void print(){
    // std::cout << "a*x" << a*x << ", a*x + b = " << a*x + b << ", ((a*x + b) \% p) = " << ((a*x + b) % p) << ", ((a*x + b) \% p) \% m = " << ((a*x + b) % p) % m << std::endl;
    std::cout << "a: " << a << ", b: " << b << ", p: " << p << ", m: " << m << std::endl;
  }
};


// TODO: make a compile-time perfect that parameterizes everything with templates 
// Hashtable comprised of perfect minimal hash functions built from linear congruential generators 
template< typename K, typename V = K, typename F = float > 
struct LCG_PMF { 
  using key_type = K; 
  const uint32_t m;        // table size 
  const uint16_t k;        // number of hash functions
  const vector< F > g;     // table terms 
  const vector< int > mul; // multiplier terms
  const vector< int > add; // additions terms
  const vector< int > mod; // modulo terms
  
  LCG_PMF(const uint32_t _m, const uint16_t _k, vector< F > G, vector< int > M, vector< int > A, vector< int > U) 
    : m(_m), k(_k), g(G), mul(M), add(A), mod(U) { }

  // Example expansion: 'g[((170*x+140)%433)%350]+g[((427*x+593)%673)%350]+g[((584*x+331)%593)%350]+g[((322*x+376)%503)%350]+g[((89*x+32)%367)%350]'
  auto operator[](K x) -> uint32_t {
    F v = 0; 
    for (uint16_t i = 0; i < k; ++i){
      v += g[((mul[i]*x + add[i]) % mod[i]) % m];
    }
    return uint32_t(v);
  }
};

template< std::integral K >
struct LcgDagHash {
  const array< uint32_t, 2 > a;  
  const array< uint32_t, 2 > b;  
  const array< uint32_t, 2 > p;   
  const uint32_t m;  // table size 
  const vector< uint32_t > g; 
  // LcgDagHash(){};  
  LcgDagHash(vector< uint32_t > _g,
    uint32_t _a1, uint32_t _a2, 
    uint32_t _b1, uint32_t _b2, 
    uint32_t _p1, uint32_t _p2,
    uint32_t _m
  ) : g(_g), a{_a1, _a2}, b{_b1, _b2}, p{_p1, _p2}, m(_m)
  { }
  
  [[nodiscard]]
  constexpr uint32_t operator()(K x) const noexcept {
    // return 0; 
    // return (((a*x + b) % p) % m) + (((a*x + b) % p) % m) % m;
    return (g[(((a[0]*x + b[0]) % p[0]) % m)] + g[(((a[1]*x + b[1]) % p[1]) % m)]) % m;
  }
};

constexpr uint32_t mod(int32_t n, int32_t m){
  return ((n % m) + m) % m;
}

template< std::integral K >
struct PerfectHashDAG {
  using node_t = pair< uint32_t, uint32_t >; 
  vector< vector< node_t > > adj;
  vector< uint32_t > vertex_values; 
  
  PerfectHashDAG(){
    srand(1234);
    adj = vector< vector< node_t > >();
    // g[((42*x+246)%397)%13]+g[((83*x+137)%353)%13])%13
  }

  void reset(){
    for (size_t i = 0; i < adj.size(); ++i){ adj[i].clear(); }
  }
  void resize(const size_t N){
    adj.resize(N);
  }

  template< typename InputIt, typename H > 
  requires IntegerHashFunction< H, K >
  void connect_all(InputIt k_it, const InputIt k_end, H f1, H f2){
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
          vertex_values.at(neighbor) = mod(int32_t(edge_value) - int32_t(vertex_values.at(vertex)), N);
        }
      }
    }
    return true; // success
  }

  template< typename InputIt >
  auto build_hash(InputIt k_it, const InputIt k_end, float mult_max, size_t n_tries, size_t n_prime, bool verbose = true) -> std::optional< LcgDagHash< K > > {
    const uint32_t N = std::distance(k_it, k_end);
    size_t n_attempts = 0; 
    bool success = false; 
    float step_sz = (ceil(mult_max*N)-N)/n_tries;
    std::vector< uint32_t > primes; 
    gen_primes_above(N, n_prime, primes);
    // std::cout << "generated primes" << std::endl;

    auto f1 = LCG(), f2 = LCG();
    // std::cout << "Configured LCGs" << std::endl;
    for (size_t i = 0; i < n_tries; ++i){
      uint32_t NG = uint32_t(N + i*step_sz);
      
      f1.randomize(NG, primes);
      f2.randomize(NG, primes);
      // std::cout << "Randomized" << std::endl;
      reset();
      resize(NG);
      // std::cout << "Connecting (" << std::distance(k_it, k_end) << ")" << std::endl;
      connect_all(k_it, k_end, f1, f2);

      // std::cout << "Assigning" << std::endl;
      if ((success = assign_values(NG))){
        if (verbose){
          std::cout << "Success @ iteration " << i << "w/ size " << NG << std::flush << std::endl;
          f1.print();
          f2.print();
        }
        // std::cout << "g[ "; for (auto ge: g){ std::cout << ge << ", ";}; std::cout << "]" << std::flush << std::endl;
        break; 
      }
    }

    // std::cout << "Success? " << success << std::endl;
    if (success){  
      for (uint32_t i = 0; k_it != k_end; ++k_it, ++i){
        auto key = *k_it;
        // std::cout << "h[" << key << "] -> " << ((g[f1(key)] + g[f2(key)]) % NG) << std::endl;
        assert(i == ((g[f1(key)] + g[f2(key)]) % NG));
      }
      return std::make_optional(LcgDagHash< K >(vertex_values, f1.a, f2.a, f1.b, f2.b, f1.p, f2.p, f1.m));
    } else {
      return std::nullopt;
    }
  }
};
