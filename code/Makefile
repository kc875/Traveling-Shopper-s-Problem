CC = g++
CU = nvcc

all: tsp_c

tsp_c: tsp_cuda.cu
	$(CU) tsp_cuda.cu -o tsp_c

tsp_s: tsp_seq.cpp
	$(CC) tsp_seq.cpp -o tsp_s

.PHONY: runcu, runse
run: runcu runse

runcu: tsp_c
	./tsp_c

runse: tsp_s
	./tsp_s

clean:
	rm tsp_c tsp_s