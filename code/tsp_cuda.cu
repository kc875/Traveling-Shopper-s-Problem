/* CUDA and sequential implemention of the Traveling Shopper Problem*/
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_ITEMS 6
#define NUM_NODES 18
#define NUM_NODES_PER_ITEM 3
#define DEBUG false

#define GRID_SIZE 136 //based on 2*SM(68 for 2080 ti)
#define BLOCK_SIZE 1024 //maximum for 2080 ti

using namespace std;





////////////////////////////////////
//        HELPER FUNCTIONS        //
////////////////////////////////////
bool rev(int i, int j) {
    return (i>j);
}


int factorial(int i) {
    if(i == 0) return 0;
    if(i == 1) return 1;
    return i * factorial(i-1);
}


double time_me(timeval start) {
    timeval end;
    gettimeofday(&end, NULL);

    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;

    return time_taken;
}


void print_arr(int * arr, int len = NUM_ITEMS) {
    for (int i = 0; i < len; i++) {
        if (arr[i]!=-1) cout<<arr[i]<<' ';
    }
    cout<<endl;
}


bool contains(vector<int> arr, int num) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == num) return true;
    }
    return false;
}


bool contains_arr(vector<vector<int>> arr, vector<int> l) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i].size() != l.size()) {
            break;
        }

        bool match = true;
        for (int j = 0; j < arr[i].size(); j++) {
            if (arr[i][j] != l[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}



vector<vector<int>> generate_permutations(vector<int> list) {
    vector<vector<int>> output(0);
    if (list.size() == 0) {
        return output;
    }
    if (list.size() == 1 || list[0] == -1) {
        output.push_back(list);
        return output;
    }
    for (int i = 0; i < list.size(); i++) {
        int start = list[i];
        if (start != -1) {
            vector<int> t(0);
            for (int j = 0; j < list.size(); j++) {
                t.push_back(list[j]);
            } 
            t.erase(t.begin()+i);

            vector<vector<int>> temp = generate_permutations(t);
            for (int j = 0; j < temp.size(); j++) {
                t.clear();
                t.push_back(start);
                for (int k = 0; k < temp[j].size(); k++) {
                    t.push_back(temp[j][k]);
                }
                output.push_back(t);
            }
        }
    }
    return output;
}




////////////////////////////////////
//           CUDA STUFF           //
////////////////////////////////////
///////////////////////
// PROBLEM GENERATION//
///////////////////////
__global__ void c_adjacency_matrix(int* out, int size, int seed) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    
    if (tidx < tidy) {
        curandState_t state;
        curand_init(seed, /* the seed controls the sequence of random values that are produced */
            tidx*size+tidy, /* the sequence number is only important with multiple cores */
            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            &state);
        int val = curand(&state)%10+1;
        *(out + tidx*size + tidy) = val;
        *(out + tidy*size + tidx) = val;
    }
    else if (tidx == tidy) {
        *(out + tidx*size + tidy) = 0;
    }
}



__global__ void c_generate_items(int *out, int nodes, int items, int nodes_per_item, int seed) {
    // 2 shops per item
    int tid = threadIdx.x;
    for (int i = 0; i < nodes_per_item; i++) {
        out[nodes_per_item*tid+i] = 0;
    }


    curandState_t state;
    curand_init(seed, /* the seed controls the sequence of random values that are produced */
        tid, /* the sequence number is only important with multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state);
    
    //generating
    for (int i = 0; i < nodes_per_item; i++) {
        int r = (curand(&state) % (nodes-1)) + 1;
        bool same = false;
        do {
            same = false;
            for (int j = i-1; j >= 0; j--) {
                if (out[nodes_per_item*tid+j] == r) {
                    same = true;
                    r = (curand(&state) % (nodes-1)) + 1;
                    break;
                }
            }
        } while (same);

        out[nodes_per_item*tid+i] = r;
    }

}





///////////////////////
//  PATH GENERATION  //
///////////////////////
__global__ void c_generate_paths(int* item_list, int * out, int nodes, int items, int nodes_per_item) {
    int tid = threadIdx.x;

    int num = tid;
    for (int i = items-1; i >= 0; i--) {
        int j = (items-1) - i;
        int choice = num/(int)(pow(nodes_per_item,i));
        out[tid*items + j] = item_list[nodes_per_item*j + choice];
        num = num%((int)pow(nodes_per_item,i));
    }

}



__device__ void c_permutation(int * path, int * output, int len, int count) {
    for (int i = 0; i<len; i++) {
        output[i] = path[i];
    }
}

__global__ void c_path_permutation(int n, int * paths, int *o_arr, int size, int * shuffle) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { 
        for (int j = 0; j < size * NUM_ITEMS; j++) {
            o_arr[i*size*NUM_ITEMS+j] = -1;
        }
        //TODO: fill output with permutations of array
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < NUM_ITEMS; k++) {
                o_arr[i*size*NUM_ITEMS+j*NUM_ITEMS+k] = paths[i*NUM_ITEMS+shuffle[j*NUM_ITEMS+k]];
            }
        }
        /*for(int j = 0; j < size; j++){
            for(int k = 0; k < NUM_ITEMS; k++){
                o_arr[i*size*NUM_ITEMS+j*NUM_ITEMS+k] = paths[i*NUM_ITEMS+k];
            }
        } */

        //TODO: check output and remove invalid permutations
        for (int j = 0; j < size; j++) {
            bool valid = true;
            bool neg = true;
            for (int k = NUM_ITEMS-1; k >= 0; k--) {
                if (o_arr[i*size*NUM_ITEMS+j*NUM_ITEMS+k]==-1) {
                    if (!neg) {
                        valid = false;
                    }
                }
                else {
                    neg = false;
                }
            }
            if (!valid) {
                //TODO: erase row (make all -1)
                for (int k = 0; k < NUM_ITEMS; k++) {
                    o_arr[i*size*NUM_ITEMS+j*NUM_ITEMS+k] = -1;
                }
            }
        }
    }
}





///////////////////////
//    PATH LENGTHS   //
///////////////////////
__global__ void c_path_length(int n, int * paths, int *adjacency, int *o_arr, int start_end = 0) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        o_arr[i] = 0;
        if (paths[i*NUM_ITEMS]==-1) {
            o_arr[i] = 10000000;         
        }
        else { 
            o_arr[i] += adjacency[start_end*NUM_NODES + paths[i*NUM_ITEMS+0]];
            int j = 0;
            for (j = 0; j < NUM_ITEMS-1; j++) {
                if (paths[i*NUM_ITEMS+j+1]==-1) {
                    break;          
                }
                o_arr[i] += adjacency[paths[i*NUM_ITEMS+j]*NUM_NODES + paths[i*NUM_ITEMS+j+1]];
            }
            o_arr[i] += adjacency[paths[i*NUM_ITEMS+j]*NUM_NODES + start_end];
        }
    }
}





///////////////////////
//   SHORTEST PATH   //
///////////////////////
__global__ void block_reduce_min(int n, int *i_arr, int *o_arr, int *o_index, bool first = true) {
    int tid = threadIdx.x;
    int sid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    const int iter = BLOCK_SIZE * gridDim.x;
  
    int min;
    int min_i;
    bool set = false;
    for (int i = sid; i < n; i += iter) {
        if (!set) {
            min = i_arr[i];
            min_i = i;
            set = true;
        }
        if (min > i_arr[i]) {
            min = i_arr[i];
            min_i = i;
        }
    }

    __shared__ int s_arr[BLOCK_SIZE];
    __shared__ int s_index[BLOCK_SIZE]; 
    s_arr[tid] = min;
    s_index[tid] = min_i;
    __syncthreads();
  
    for (int size = BLOCK_SIZE/2; size > 0; size /= 2) {
        if (tid < size) {
            if(s_arr[tid] > s_arr[tid + size]) {
                s_arr[tid] = s_arr[tid + size]; 
                s_index[tid] = s_index[tid + size];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        o_arr[blockIdx.x] = s_arr[0];
        if(first){
            o_index[blockIdx.x] = s_index[0];
        }
        else{
            o_index[blockIdx.x] = o_index[s_index[0]];
        }
    }
}



__global__ void c_find_min(int size, int * c_lengths, int * c_out, int * c_index) {
    int thread = threadIdx.x;
    int num_threads = blockDim.x;

    int min = 100000;
    int id = -1;

    for (int i = thread; i < size; i += num_threads) {
        if (c_lengths[i] < min && c_lengths[i] > 0) {
            min = c_lengths[i];
            id = i;
        }
    }

    c_out[thread] = min;
    c_index[thread] = id;
    __syncthreads();

    for (int part_size = num_threads/2; part_size > 0; part_size /= 2) {
        if (thread < part_size) {
            if (c_out[part_size + thread] < c_out[thread] && c_out[part_size + thread] > 0 && c_index[part_size + thread] > 0) {
                c_out[thread] = c_out[part_size + thread];
                c_index[thread] = c_index[part_size + thread];
            }
            __syncthreads();
        }
    }

}





///////////////////////
//   CUDA FUNCTION   //
///////////////////////  
void ts_cuda() {
    struct timeval start;

    // problem generation
    gettimeofday(&start, NULL);

    cout<<"---Starting cuda calculation---"<<endl;

    cout<<"Generating problem"<<endl;
    int * c_adjacency;
    int * c_items;
    cudaMalloc(&c_adjacency, NUM_NODES*NUM_NODES*sizeof(int));
    cudaMalloc(&c_items, NUM_NODES_PER_ITEM*NUM_ITEMS*sizeof(int));

    c_adjacency_matrix<<<1, dim3(NUM_NODES, NUM_NODES)>>>(c_adjacency, NUM_NODES, (int)time(0));

    c_generate_items<<<1, NUM_ITEMS>>>(c_items, NUM_NODES, NUM_ITEMS, NUM_NODES_PER_ITEM, (int)time(0));

    cout<<"Calculating paths"<<endl;
    // partial paths
    int num_paths = (int)pow(NUM_NODES_PER_ITEM, NUM_ITEMS);
    int * c_part_paths;
    cudaMalloc(&c_part_paths, num_paths*NUM_ITEMS*sizeof(int));
    c_generate_paths<<<1,num_paths>>>(c_items, c_part_paths, NUM_NODES, NUM_ITEMS, NUM_NODES_PER_ITEM);
    if (DEBUG) {
        int * tmp_out3 = (int*) malloc(num_paths*NUM_ITEMS*sizeof(int));
        cudaDeviceSynchronize();
        cudaMemcpy(tmp_out3, c_part_paths, num_paths*NUM_ITEMS*sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_paths; i++) {
            for (int j = 0; j < NUM_ITEMS; j++) {
                cout<<tmp_out3[i*NUM_ITEMS + j]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
        free(tmp_out3);
    }

    
    // paths
    int size = num_paths;
    int * c_paths;
    int f = factorial(NUM_ITEMS);
    int * c_shuffle;

    vector<int> t;
    for (int i = 0; i < NUM_ITEMS; i++) {
        t.push_back(i);
    }
    vector<vector<int>> temp = generate_permutations(t);
    t.clear();
    for (int i = 0; i < temp.size(); i++) {
        for (int j = 0; j < temp[i].size(); j++) {
            t.push_back(temp[i][j]);
        }
        //print_arr(&temp[i][0]);
    }
    
    cudaMalloc(&c_shuffle, f*NUM_ITEMS*sizeof(int));
    cudaMemcpy(c_shuffle, &t[0], f*NUM_ITEMS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&c_paths, num_paths*NUM_ITEMS*f*sizeof(int));

    c_path_permutation<<<(size + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(size, c_part_paths, c_paths, f, c_shuffle);
    if (DEBUG) {
        int * tmp_out2 = (int*) malloc(num_paths*NUM_ITEMS*f*sizeof(int));
        cudaDeviceSynchronize();
        cudaMemcpy(tmp_out2, c_paths, num_paths*NUM_ITEMS*f*sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_paths*f; i++) {
            for (int j = 0; j < NUM_ITEMS; j++) {
                cout<<tmp_out2[i*NUM_ITEMS + j]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
        free(tmp_out2);
    }



    cout<<"Calculate path lengths"<<endl;
    size = num_paths*f;
    int * c_lengths;
    cudaMalloc(&c_lengths, size*sizeof(int));

    c_path_length<<<(size + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(size, c_paths, c_adjacency, c_lengths);
    if (DEBUG) {
        int * tmp_out1 = (int*) malloc(size*sizeof(int));
        cudaDeviceSynchronize();
        cudaMemcpy(tmp_out1, c_lengths, size*sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            cout<<tmp_out1[i]<<" ";
        }
        cout<<endl;
        free(tmp_out1);
        cout<<size<<" paths\n"<<endl;
    }

    cudaDeviceSynchronize();

    cout<<"Finding shortest path"<<endl;

    int *c_out;
    int *c_index;
    cudaMalloc(&c_out, sizeof(int) * BLOCK_SIZE);  
    cudaMalloc(&c_index, sizeof(int) * BLOCK_SIZE);


    c_find_min<<<1,BLOCK_SIZE>>>(size, c_lengths, c_out, c_index);

    cudaDeviceSynchronize();

    int index;
    int min;
    int *path = (int*)malloc(num_paths*NUM_ITEMS*f*sizeof(int));
    
    cudaMemcpy(&index, c_index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min, c_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(path, c_paths, num_paths*NUM_ITEMS*f*sizeof(int), cudaMemcpyDeviceToHost);
    
    double etime = time_me(start);
    cout<<"Shortest Path Length: "<<min<<"\nIndex: "<<index<<endl;
    print_arr(&path[index*NUM_ITEMS]);

    cout<<"Time Taken (cuda): "<<fixed<<setprecision(6)<<etime<<"\n"<<endl;

    cudaFree(c_adjacency);
    cudaFree(c_items);
    cudaFree(c_part_paths);
    cudaFree(c_paths);
    cudaFree(c_lengths);
    cudaFree(c_out);
    cudaFree(c_index);
    free(path);
}



int main(void) {
    ts_cuda();
}
