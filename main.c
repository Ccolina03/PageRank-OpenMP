#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>   
#include <mpi.h>
#include "Lab4_IO.h" 

#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

int main(int argc, char* argv[]){
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int nodecount;
    struct node *nodehead = NULL;
    double *rankOld, *rankNew;
    double *contrib; // initial rank to add upon
    int iterationcount = 0;
    double start_time, end_time, elapsed_time;

    // Rank 0 reads the number of nodes from file and broadcasts it. 1 time operation
    if(rank == 0){
        FILE *ip = fopen("data_input_meta", "r");
        if(ip == NULL){
            printf("Error opening the data_input_meta file.\n");
            MPI_Abort(MPI_COMM_WORLD, 253);
        }
        fscanf(ip, "%d\n", &nodecount);
        fclose(ip);
    }
    MPI_Bcast(&nodecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Every process loads the entire graph.
    if(node_init(&nodehead, 0, nodecount) != 0){
        MPI_Abort(MPI_COMM_WORLD, 254);
    }

    // Allocate arrays for PageRank values (double buffering) and contributions.
    rankOld = (double*) malloc(nodecount * sizeof(double));
    rankNew = (double*) malloc(nodecount * sizeof(double));
    contrib = (double*) malloc(nodecount * sizeof(double));
    if(rankOld == NULL || rankNew == NULL || contrib == NULL){
        printf("Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize rankOld: every node gets 1/nodecount.
    for (int i = 0; i < nodecount; i++){
        rankOld[i] = 1.0 / nodecount;
    }

    // Determine the local range of nodes for each process.
    int local_start, local_count;
    int base_nodes = nodecount / num_procs;
    int rem = nodecount % num_procs;
    if(rank < rem){
        local_start = rank * (base_nodes + 1);
        local_count = base_nodes + 1;
    } else {
        local_start = rank * base_nodes + rem;
        local_count = base_nodes;
    }
    int local_end = local_start + local_count;

    // Allocate a temporary send buffer for the local segment.
    double *local_send = (double*) malloc(local_count * sizeof(double));
    if(local_send == NULL){
        printf("Temporary send buffer allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Prepare arrays for MPI_Allgatherv
    int *recvcounts = (int*) malloc(num_procs * sizeof(int));
    int *displs = (int*) malloc(num_procs * sizeof(int));
    if(recvcounts == NULL || displs == NULL){
        printf("Allocation error for recvcounts/displs.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int p = 0; p < num_procs; p++){
        int base_p = nodecount / num_procs;
        int rem_p = nodecount % num_procs;
        if(p < rem_p){
            recvcounts[p] = base_p + 1;
            displs[p] = p * (base_p + 1);
        } else {
            recvcounts[p] = base_p;
            displs[p] = p * base_p + rem_p;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    double global_error, global_diff, global_norm;
    const double base_val = (1.0 - DAMPING_FACTOR) / nodecount;
    const double d = DAMPING_FACTOR;
    MPI_Request request;

    do{
        iterationcount++;

        // Precompute contributions for each node using rankOld.
        for (int j = 0; j < nodecount; j++){
            if(nodehead[j].num_out_links > 0)
                contrib[j] = rankOld[j] / nodehead[j].num_out_links;
            else
                contrib[j] = 0.0;
        }

        // Update local nodes: compute new PageRank values for indices [local_start, local_end).
        for (int i = local_start; i < local_end; i++){
            double new_val = base_val;
            for (int k = 0; k < nodehead[i].num_in_links; k++){
                int j = nodehead[i].inlinks[k]; // global index of a node linking to i.
                new_val += d * contrib[j];
            }
            rankNew[i] = new_val;
        }

        // Copy the local updated segment to the temporary send buffer.
        memcpy(local_send, &rankNew[local_start], local_count * sizeof(double));

        // Initiate non-blocking gather using the separate local_send buffer.
        MPI_Iallgatherv(local_send, local_count, MPI_DOUBLE,
                        rankNew, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD, &request);

        // Compute error for the local segment (using current rankOld and rankNew values).
        double local_diff = 0.0, local_norm = 0.0;
        for (int i = local_start; i < local_end; i++){
            local_diff += fabs(rankNew[i] - rankOld[i]);
            local_norm += fabs(rankNew[i]);
        }

        // Wait for the non-blocking gather to complete so that rankNew is fully updated.
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // Aggregate error across all processes.
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_error = global_diff / global_norm;

        // Swap pointers instead of copying arrays.
        double *temp = rankOld;
        rankOld = rankNew;
        rankNew = temp;

    } while(global_error >= EPSILON);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // After the final iteration, rankOld holds the latest PageRank values.
    if(rank == 0){
        Lab4_saveoutput(rankOld, nodecount, elapsed_time);
    }

    // Cleanup.
    free(recvcounts);
    free(displs);
    free(contrib);
    free(local_send);
    node_destroy(nodehead, nodecount);
    free(rankOld);
    free(rankNew);

    MPI_Finalize();
    return 0;
}
