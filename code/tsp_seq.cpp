/* serial implementation of the Traveling Shopper's Problem */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <algorithm>

#define NUM_ITEMS 6
#define NUM_NODES 18
#define NUM_NODES_PER_ITEM 3
#define DEBUG false

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
//          SERIAL STUFF          //
////////////////////////////////////
///////////////////////
// PROBLEM GENERATION//
///////////////////////
void generate_adjacency_matrix (int *out, int size) {
    srand(time(0));
    for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
            if (i == j) {
                out[i*size+j] = 0;
            } else {
                int r = rand() % 100 + 10;
                out[i*size+j] = r;
                out[j*size+i] = r;
            }
        }
    }
}



void generate_items(int * out, int nodes, int items, int nodes_per_item) {
    srand(time(0));

    cout<<"randoms: "<<endl;
    for (int i = 0; i < items; i++) {
        for (int j = 0; j < nodes_per_item; j++) {
            int r = (rand() % (nodes-1)) + 1;
            bool same = false;
            do {
                same = false;
                for (int k = j-1; k >= 0; k--) {
                    if (out[nodes_per_item*i+k] == r) {
                        same = true;
                        r = (rand() % (nodes-1)) + 1;
                        break;
                    }
                }
            } while (same);
            out[nodes_per_item*i+j] = r;
        }
    }
}





///////////////////////
//  PATH GENERATION  //
///////////////////////
vector<int> generate_paths(int * item_list, int nodes, int items, int nodes_per_item) {
    struct timeval start;
    gettimeofday(&start, NULL);
    vector<vector<int>> paths(0);
    vector<vector<int>> temp(0);
    int num_paths = (int)pow(NUM_NODES_PER_ITEM, NUM_ITEMS);
    for (int ni = 0; ni < num_paths; ni++) {
        vector<int> t(0);
        int num = ni;
        for (int i = items-1; i >= 0; i--) {
            int j = (items-1) - i;
            int choice = num/(int)(pow(nodes_per_item,i));
            t.push_back(item_list[nodes_per_item*j + choice]);
            num = num%((int)pow(nodes_per_item,i));
        }
        if (!contains_arr(paths,t)) {
            paths.push_back(t);
        }
    }

    cout<<"Time Taken: "<<fixed<<setprecision(6)<<time_me(start)<<"\n"<<endl;

    cout<<"Calculating "<<paths.size()<<" permutations"<<endl;
    //paths now hold every combination of nodes
    while (paths.size() > 0) {
        vector<int> t = paths.back();
        paths.pop_back();
        vector<vector<int>> temp2 = generate_permutations(t);
        if (DEBUG) {
            cout<<"permutations"<<endl;
            for (int i = 0; i < t.size(); i++) {
                cout<<t[i]<<' ';
            }
            cout<<"\n---"<<endl;
        }  
        for (int i = 0; i < temp2.size(); i++) {
            temp.push_back(temp2[i]);
        }
    }
    cout<<"Done calculating permutations"<<endl;
    vector<int> output(0);
    for (int i = 0; i < temp.size(); i++) {
        for (int j = 0; j < temp[i].size(); j++) {
            output.push_back(temp[i][j]);
        }
    }
    return output;
}






///////////////////////
//    PATH LENGTHS   //
///////////////////////
int calculate_path_length(int * path, int * adjacency, int start_end = 0, int len = NUM_ITEMS) {
    int out = 0;

    if (path[0]==-1) {
        out = 10000000;         
    }
    else { 
        out += adjacency[start_end*NUM_NODES + path[0]];
        int j = 0;
        for (j = 0; j < NUM_ITEMS-1; j++) {
            if (path[j+1]==-1) {
                break;          
            }
            out += adjacency[path[j]*NUM_NODES + path[j+1]];
        }
        out += adjacency[path[j]*NUM_NODES + start_end];
    }

    return out;
}





///////////////////////
//   SHORTEST PATH   //
///////////////////////
void calculate_lengths(int * lengths, vector<int> paths, int * adjacency, int start_end = 0) {
    for (int i = 0; i < (paths.size()/NUM_ITEMS); i++) {
        lengths[i] = calculate_path_length(&paths[i*NUM_ITEMS], adjacency);
    }
}




///////////////////////
//  SERIAL FUNCTION  //
///////////////////////  
void ts_serial(){
    struct timeval start;
    gettimeofday(&start, NULL);

    cout<<"---Starting serial calculation---"<<endl;
    int *adjacency = (int*)malloc(NUM_NODES * NUM_NODES * sizeof(int));
    int *items = (int*)malloc(NUM_NODES_PER_ITEM * NUM_ITEMS * sizeof(int));
    for (int i = 0; i < NUM_ITEMS; i++) {
        for (int j = 0; j < NUM_NODES_PER_ITEM; j++) {
            cout<<items[i*NUM_NODES_PER_ITEM + j]<<"\t";
        }
        cout<<endl;
    }

    cout<<"Generating adjacency matrix"<<endl;
    generate_adjacency_matrix(adjacency, NUM_NODES);
    cout<<"Generating shops list"<<endl;
    generate_items(items, NUM_NODES, NUM_ITEMS, NUM_NODES_PER_ITEM);

    cout<<"Generated lists time: "<<time_me(start)<<"\n"<<endl;


    cout<<"Calculating paths"<<endl;
    vector<int> paths = generate_paths(items, NUM_NODES, NUM_ITEMS, NUM_NODES_PER_ITEM);
    cout<<"Calculated "<<(paths.size()/NUM_NODES)<<" paths"<<endl;

    double ptime = time_me(start);
    cout<<"Time Taken: "<<fixed<<setprecision(6)<<ptime<<"\n"<<endl;

    cout<<"Calculate path lengths"<<endl;
    int * lengths = (int*)malloc(paths.size()/NUM_ITEMS*sizeof(int));
    calculate_lengths(lengths, paths, adjacency);

    cout<<"Time Taken (serial): "<<fixed<<setprecision(6)<<time_me(start)<<"\n"<<endl;

    cout<<"Finding shortest path"<<endl;
    int index = 0;
    int min = lengths[0];
    for (int i = 0; i < (paths.size()/NUM_ITEMS); i++) {
        if (min > lengths[i]) {
            index = i;
            min = lengths[i];
        }
    }

    cout<<"Shortest path: "<<lengths[index]<<endl;
    print_arr(&paths[index*NUM_ITEMS]);

    double etime = time_me(start);
    cout<<"Time minus path calculation: "<<fixed<<setprecision(6)<<(etime - ptime)<<endl;
    cout<<"Time Taken (serial): "<<fixed<<setprecision(6)<<etime<<"\n"<<endl;

    free(lengths);
}

int main(void) {
    ts_serial();
}