template <typename F, typename T>
concept HashFunction = std::regular_invocable<F, T> && std::convertible_to< std::invoke_result_t< F, T >, size_t >; 

template < typename Table, typename Key, typename Value >
concept HashTable = requires(Table h) {
  std::unsigned_integral< Key >;
  std::unsigned_integral< Value >;
  { h.operator()() } -> std::convertible_to< size_t >;
};


// TODO: make a compile-time perfect that parameterizes everything with templates 
// Hashtable comprised of perfect minimal hash functions built from linear congruential generators 
template< typename K, typename F = float > 
struct LCG_PMF { 
  // 'g[((170*x+140)%433)%350]+g[((427*x+593)%673)%350]+g[((584*x+331)%593)%350]+g[((322*x+376)%503)%350]+g[((89*x+32)%367)%350]'
  const uint32_t m; // table size 
  const uint16_t k; // number of hash functions
  const vector< F > g;     // table terms 
  const vector< int > mul; // multiplier terms
  const vector< int > add; // additions terms
  const vector< int > mod; // modulo terms
  
  // LCG_PMF(const uint32_t _m, const uint16_t _k, vector< int > M, vector< int > A, vector< int > U) : mul(M), add(A), mod(U) {}
  auto operator()(K x) -> uint32_t {
    F v = 0; 
    for (uint16_t i = 0; i < k; ++i){
      v += g[((mul[i]*x + add[i]) % mod[i]) % m];
    }
    return uint32_t(v);
  }
}