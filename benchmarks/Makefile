

laplacian:
	clang++ profile_laplacian.cpp -O2 -std=c++17 -g -o profile_laplacian -lprofiler

profile: 
	LD_PRELOAD=/usr/local/lib/libprofiler.dylib CPUPROFILE=main.prof CPUPROFILE_FREQUENCY=100000 ./profile_laplacian

report: 
	pprof --text main.prof > profile_results.txt