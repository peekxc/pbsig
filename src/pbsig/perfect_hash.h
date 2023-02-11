#include <vector>
#include <algorithm>
using std::vector; 
using uint_t = std::uint32_t;

// struct PerfectHash {
//   const vector< uint_t > v;
//   vector< uint_t > g;
//   PerfectHash(const vector< uint_t >& keys) : v(keys) {
//     g = vector< uint_t >(max_size);
//   }
//   void build() {
//     const size_t n = v.size();
//     const size_t m = n * 2;         // pigeonhole principle
//     vector< uint_t > h(m, -1);      // temporary needed to build  
//     for (size_t i = 0; i < n; ++i) {
//       uint_t x = v[i];
//       uint_t j = x % m;
//       uint_t k = 1 + (x % (m - 1)); // prime number 
//       while (h[j] != -1) {
//         j = (j + k) % m;
//       }
//       h[j] = i;
//       g[i] = j;
//     }
//   }

//   [[nodiscard]]
//   auto operator[](uint_t key) const {
//     return g[key];
//   }
// };