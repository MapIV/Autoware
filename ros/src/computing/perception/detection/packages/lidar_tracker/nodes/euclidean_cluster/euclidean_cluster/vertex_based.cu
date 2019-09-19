#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include <cuda.h>

#define TEST_VERTEX_ 1

extern __shared__ float local_buff[];

__global__ void frontierInitialize(int *frontier_array, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		frontier_array[i] = 1;
	}
}

__global__ void countAdjacentList(float *x, float *y, float *z, int point_num, float threshold, long long *adjacent_count)
{
	float *local_x = local_buff;
	float *local_y = local_x + blockDim.x;
	float *local_z = local_y + blockDim.x;
	int pid;
	int last_point = (point_num / blockDim.x) * blockDim.x;	// Exclude the last block
	float dist;

	for (pid = threadIdx.x + blockIdx.x * blockDim.x; pid < last_point; pid += blockDim.x * gridDim.x) {
		float tmp_x = x[pid];
		float tmp_y = y[pid];
		float tmp_z = z[pid];
		int count = 0;

		int block_id;

		for (block_id = 0; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
				count += (i + block_id != pid && dist < threshold) ? 1 : 0;
			}
			__syncthreads();
		}

		__syncthreads();

		// Compare with last block
		if (threadIdx.x < point_num - block_id) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
		}
		__syncthreads();

		for (int i = 0; i < point_num - block_id; i++) {
			dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
			count += (i + block_id != pid && dist < threshold) ? 1 : 0;
		}

		adjacent_count[pid] = count;
		__syncthreads();
	}
	__syncthreads();


	// Handle last block
	if (pid >= last_point) {
		int count = 0;
		float tmp_x, tmp_y, tmp_z;

		if (pid < point_num) {
			tmp_x = x[pid];
			tmp_y = y[pid];
			tmp_z = z[pid];
		}

		__syncthreads();

		int block_id;

		for (block_id = 0; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			if (pid < point_num) {
				for (int i = 0; i < blockDim.x; i++) {
					dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
					count += (i + block_id != pid && dist < threshold) ? 1 : 0;
				}
			}
			__syncthreads();
		}
		__syncthreads();

		if (pid < point_num) {
			local_x[threadIdx.x] = x[pid];
			local_y[threadIdx.x] = y[pid];
			local_z[threadIdx.x] = z[pid];
			__syncthreads();

			for (int i = 0; i < point_num - block_id; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
				count += (i + block_id != pid && dist < threshold) ? 1 : 0;
			}
			__syncthreads();

			adjacent_count[pid] = count;
		}
	}
}

__global__ void buildAdjacentList(float *x, float *y, float *z, int point_num, float threshold, long long *adjacent_count, int *adjacent_list)
{
	float *local_x = local_buff;
	float *local_y = local_x + blockDim.x;
	float *local_z = local_y + blockDim.x;
	int pid;
	int last_point = (point_num / blockDim.x) * blockDim.x;

	for (pid = threadIdx.x + blockIdx.x * blockDim.x; pid < last_point; pid += blockDim.x * gridDim.x) {
		long long writing_location = adjacent_count[pid];
		float tmp_x = x[pid];
		float tmp_y = y[pid];
		float tmp_z = z[pid];

		int block_id;

		for (block_id = 0; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				if (i + block_id != pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
					adjacent_list[writing_location++] = i + block_id;
				}
			}
			__syncthreads();
		}
		__syncthreads();


		if (threadIdx.x < point_num - block_id) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
		}
		__syncthreads();

		for (int i = 0; i < point_num - block_id; i++) {
			if (i + block_id != pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
				adjacent_list[writing_location++] = i + block_id;
			}
		}
		__syncthreads();

	}

	if (pid >= last_point) {
		float tmp_x, tmp_y, tmp_z;
		int writing_location;

		if (pid < point_num) {
			tmp_x = x[pid];
			tmp_y = y[pid];
			tmp_z = z[pid];
			writing_location = adjacent_count[pid];
		}

		int block_id;

		for (block_id = 0; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			if (pid < point_num) {
				for (int i = 0; i < blockDim.x; i++) {
					if (i + block_id != pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
						adjacent_list[writing_location++] = i + block_id;
					}
				}
			}
			__syncthreads();
		}
		__syncthreads();

		if (pid < point_num) {
			local_x[threadIdx.x] = x[pid];
			local_y[threadIdx.x] = y[pid];
			local_z[threadIdx.x] = z[pid];
			__syncthreads();

			for (int i = 0; i < point_num - block_id; i++) {
				if (i + block_id != pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
					adjacent_list[writing_location++] = i + block_id;
				}
			}
		}
	}
}

__global__ void clustering(long long *adjacent_list_loc, int *adjacent_list, int point_num, int *cluster_name, int *frontier_array1, int *frontier_array2, bool *changed)
{
	__shared__ bool schanged;

	if (threadIdx.x == 0)
		schanged = false;
	__syncthreads();

	for (int pid = threadIdx.x + blockIdx.x * blockDim.x; pid < point_num; pid += blockDim.x * gridDim.x) {
		if (frontier_array1[pid] == 1) {
			frontier_array1[pid] = 0;
			int cname = cluster_name[pid];
			bool c = false;
			long long start = adjacent_list_loc[pid];
			long long end = adjacent_list_loc[pid + 1];

			// Iterate through neighbors' ids
			for (long long i = start; i < end; i++) {
				int nid = adjacent_list[i];
				int nname = cluster_name[nid];
				if (cname < nname) {
					atomicMin(cluster_name + nid, cname);
					frontier_array2[nid] = 1;
					schanged = true;
					//*changed = true;
				} else if (cname > nname) {
					cname = nname;
					c = true;
				}
			}

			if (c) {
				atomicMin(cluster_name + pid, cname);
				frontier_array2[pid] = 1;
				schanged = true;
				//*changed = true;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && schanged)
		*changed = true;
}

/* Iterate through the list of remaining clusters and mark the corresponding
 * location on cluster location array by 1
 */
__global__ void clusterMark2(int *cluster_list, int *cluster_location, int cluster_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = idx; i < cluster_num; i += blockDim.x * gridDim.x) {
		cluster_location[cluster_list[i]] = 1;
	}
}

void GpuEuclideanCluster2::extractClusters2()
{
	long long total_time, build_graph, clustering_time;
	int iteration_num;

	extractClusters2(total_time, build_graph, clustering_time, iteration_num);
}



void GpuEuclideanCluster2::extractClusters2(long long &total_time, long long &build_graph, long long &clustering_time, int &iteration_num)
{
#ifdef DEBUG_
	std::cout << "VERTEX-BASED 1: compare pairwise distance of all points" << std::endl;
#endif

	total_time = build_graph = clustering_time = 0;
	iteration_num = 0;

	initClusters();

	int block_x = (point_num_ < block_size_x_) ? point_num_ : block_size_x_;
	int grid_x = (point_num_ - 1) / block_x + 1;

	long long *adjacent_count;
	int *adjacent_list;

	struct timeval start, end;

	gettimeofday(&start, NULL);

	checkCudaErrors(cudaMalloc(&adjacent_count, sizeof(long long) * (point_num_ + 1)));

	countAdjacentList<<<grid_x, block_x, sizeof(float) * block_size_x_ * 3>>>(x_, y_, z_, point_num_, threshold_, adjacent_count);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Count ADJ = " << timeDiff(start, end) << std::endl;
#endif

	long long adjacent_list_size;

	gettimeofday(&start, NULL);


	GUtilities::exclusiveScan(adjacent_count, point_num_ + 1, &adjacent_list_size);

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

	if (adjacent_list_size == 0) {
		checkCudaErrors(cudaFree(adjacent_count));
		cluster_num_ = 0;
		return;
	}

	gettimeofday(&start, NULL);

	std::cout << "Adjacent list size = " << adjacent_list_size << std::endl;
	checkCudaErrors(cudaMalloc(&adjacent_list, sizeof(int) * adjacent_list_size));

	buildAdjacentList<<<grid_x, block_x, sizeof(float) * block_size_x_ * 3>>>(x_, y_, z_, point_num_, threshold_, adjacent_count, adjacent_list);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Build ADJ = " << timeDiff(start, end) << std::endl;
#endif

	bool *changed;

	bool hchanged;
	checkCudaErrors(cudaMalloc(&changed, sizeof(bool)));

	int *frontier_array1, *frontier_array2;

#ifdef TEST_VERTEX_
	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaMalloc(&frontier_array1, sizeof(int) * point_num_));
	checkCudaErrors(cudaMalloc(&frontier_array2, sizeof(int) * point_num_));

	frontierInitialize<<<grid_x, block_x>>>(frontier_array1, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemset(frontier_array2, 0, sizeof(int) * point_num_));
	checkCudaErrors(cudaDeviceSynchronize());

	int itr = 0;

	do {
		hchanged = false;
		checkCudaErrors(cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice));

		clustering<<<grid_x, block_x>>>(adjacent_count, adjacent_list, point_num_, cluster_name_, frontier_array1, frontier_array2, changed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		int *tmp;

		tmp = frontier_array1;
		frontier_array1 = frontier_array2;
		frontier_array2 = tmp;

		checkCudaErrors(cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost));

		itr++;
	} while (hchanged);


	gettimeofday(&end, NULL);

	clustering_time += timeDiff(start, end);
	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Iteration = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;
#endif

	iteration_num = itr;

	// renaming clusters
	int *cluster_location;

	gettimeofday(&start, NULL);
	checkCudaErrors(cudaMalloc(&cluster_location, sizeof(int) * (point_num_ + 1)));
	checkCudaErrors(cudaMemset(cluster_location, 0, sizeof(int) * (point_num_ + 1)));

	clusterMark2<<<grid_x, block_x>>>(cluster_name_, cluster_location, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::exclusiveScan(cluster_location, point_num_ + 1, &cluster_num_);

	renamingClusters(cluster_name_, cluster_location, point_num_);

	checkCudaErrors(cudaFree(adjacent_count));
	checkCudaErrors(cudaFree(adjacent_list));
	checkCudaErrors(cudaFree(frontier_array1));
	checkCudaErrors(cudaFree(frontier_array2));
	checkCudaErrors(cudaFree(changed));
	checkCudaErrors(cudaFree(cluster_location));
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

#ifndef DEBUG_
	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;
#endif
}
