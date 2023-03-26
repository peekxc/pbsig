CC=/usr/local/opt/llvm/bin/clang
CXX=/usr/local/opt/llvm/bin/clang++

clang_version: 
	$(CXX) --version

laplacian:
	$(CXX) profile_laplacian.cpp -O2 -std=c++20 -g -o profile_laplacian -L/usr/local/lib/ -lprofiler -I/Users/mpiekenbrock/pbsig/src/pbsig/

iterators: 
	$(CXX) profile_iter.cpp -O2 -std=c++20 -g -o profile_iter -L/usr/local/lib -lprofiler -I/Users/mpiekenbrock/pbsig/src/pbsig/

test_pmh: test_pmh.cpp
	$(CXX) test_pmh.cpp -O2 -std=c++20 -g -o test_pmh -L/usr/local/lib -lprofiler -I/Users/mpiekenbrock/pbsig/src/pbsig/

profile_laplacian: 
	LD_PRELOAD=/usr/local/lib/libprofiler.dylib CPUPROFILE=main.prof CPUPROFILE_FREQUENCY=100000 ./profile_laplacian

profile_iterators: 
	LD_PRELOAD=/usr/local/lib/libprofiler.dylib CPUPROFILE=main.prof CPUPROFILE_FREQUENCY=100000 ./profile_iter

test_ranks: test_ranks.cpp
	$(CXX) test_ranks.cpp -O2 -std=c++20 -g -o test_ranks -L/usr/local/lib/ -I/Users/mpiekenbrock/pbsig/src/pbsig/

test_bc: test_bc.cpp
	$(CXX) test_bc.cpp -O2 -std=c++20 -g -o test_bc -L/usr/local/lib/ -I/Users/mpiekenbrock/pbsig/src/pbsig/

report: 
	pprof --text main.prof > profile_results.txt

report_pdf_iter:
	pprof --pdf main.prof > profile_iter.pdf