#include <concepts> 
#include <vector> 
#include <cinttypes>

using std::vector; 

// template < typename F, typename T >
// concept IntegerHashFunction = std::regular_invocable<F, T> && std::convertible_to< std::invoke_result_t< F, T >, size_t >; 

template < typename Table, typename Key = typename Table::key_type >
concept IntegralHashTable = requires(Table h, Key k) {
  { h.operator[](k) } -> std::integral;
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