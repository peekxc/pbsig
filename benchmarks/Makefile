CC=/usr/local/opt/llvm/bin/clang
CXX=/usr/local/opt/llvm/bin/clang++

clang_version: 
	$(CXX) --version

laplacian:
	$(CXX) profile_laplacian.cpp -O2 -std=c++20 -o profile_laplacian -L/usr/local/lib/ -lprofiler -I/Users/mpiekenbrock/pbsig/src/pbsig/

iterators: 
	$(CXX) profile_iter.cpp -O2 -std=c++20 -g -o profile_iter -L/usr/local/lib -lprofiler -I/Users/mpiekenbrock/pbsig/src/pbsig/

profile: 
	LD_PRELOAD=/usr/local/lib/libprofiler.dylib CPUPROFILE=main.prof CPUPROFILE_FREQUENCY=100000 ./profile_laplacian

profile_iterators: 
	LD_PRELOAD=/usr/local/lib/libprofiler.dylib CPUPROFILE=main.prof CPUPROFILE_FREQUENCY=100000 ./profile_iter


report: 
	pprof --text main.prof > profile_results.txt

report_pdf_iter:
	pprof --pdf main.prof > profile_iter.pdf