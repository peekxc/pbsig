
from pbsig.simplicial import Simplex, SimplicialComplex

s = Simplex([0,1,2,3])

S = SimplicialComplex([[0,1,2,3,4]])


Simplex(0)

S = SimplicialComplex([(0), (1), (0,1)])

S.remove(0)


# np.array([str(s.vertices) for s in S])
  #'{0: <5}'.format(''.join(str(s)[1:-1].split(',')))# '{0: <5}'.format(

print(*[s+'\n' for s in SC])
print(SC[0])
print(SC[1])
print(SC[2])
print(SC[3])
'{0: <5}'.format('0123')




K = SimplicialComplex([0,2,[0,2]])


S.add([0,1,2,3])


S = SimplicialComplex([(0), (1), (2), (0,1), (0,2), (1,2), (0,1,2)])
K = SimplicialComplex([(0), (2), (0,1), (0,2), (2)])


## Test filtration building
S = [Simplex(0), Simplex(1), Simplex([0,1]), Simplex([0,1,2]), Simplex([0,2]), Simplex([1,2]), Simplex([2])]

sdtype = np.dtype([('s', Simplex), ('index', int)])
 
## Yes, structured arrays are the way to go. 
w = np.sort(np.fromiter(iter(S), dtype=Simplex))

d = np.fromiter((s.dimension() for s in S), dtype=int)

from operator import itemgetter as I

I = np.repeat(0, len(S))
SI = sorted(zip(S, I), key=lambda s: (s[1], s[0].dimension(), s, tuple(s[0])))  #, s.dimension(), tuple(s), s)

# np.array(F.s, dtype=Simplex)
F = np.fromiter(iter(SI), dtype=sdtype)
F = np.rec.array(F, dtype=sdtype)

sdtype.names
sdtype.fields

from pbsig.utility import rank_combs

[s.dimension() for s in S]
Sf = np.random.uniform(size=len(S))
R = [(rank_comb(s.vertices, n=3, k=s.dimension()+1), s.dimension(), f) for s,f in zip(S, Sf)]

R = np.array(R, dtype=[(('rank','r'),np.int64), (('dim','d'), np.int8), (('index', 'i'), np.float32)])
R = np.rec.array(R)

R = R[np.argsort(R, order=['i', 'd', 'r'])]

from pbsig.utility import unrank_combs, unrank_comb


#unrank_combs(R.rank, k=R.dim, n=3)

[(Simplex(unrank_comb(r, k=d+1, n=3)), i) for r,d,i in R]

np.argsort(R, order=['r', 'd'])

w[:-1] <= w[1:]