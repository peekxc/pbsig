import timeit
timeit.timeit(lambda: lo @ x, number=150)
lo.prepare(1)
timeit.timeit(lambda: lo @ x, number=150)

import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(lo._matvec)
profile.enable_by_count()
_ = eigsh(lo, k=5, which='LM', return_eigenvectors=False)
profile.print_stats(output_unit=1e-3)

eps, p = 0.1, 1
s1 = np.vectorize(lambda t: (t**p)/(t**p + eps**p))
s2 = np.vectorize(lambda t: (t**p)/(t**p + eps))
s3 = np.vectorize(lambda t: t/((t**p + eps**p)**(1/p)))
s4 = np.vectorize(lambda t: 1 - np.exp(-t/eps))

x = np.linspace(-1, 1, 1000)

for eps in [1e-12, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 10]:
  plt.plot(x, s1(abs(x)), label=f"{eps:.3f}")
plt.gca().set_ylim(0, 1)
plt.legend()

plt.plot(x, s2(abs(x)))
plt.gca().set_ylim(0, 1)

for eps in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 10]:
  plt.plot(x, s3(abs(x)))
plt.gca().set_ylim(0, 1)


for eps in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 10]:
  plt.plot(x, s4(abs(x)))
  plt.gca().set_ylim(0, 1)