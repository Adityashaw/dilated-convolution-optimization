
#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>
#include<math.h>

#define ull unsigned long long int

__global__ void Convolute(int input_row_size, int input_col_size, int *input,
		int kernel_row_size, int kernel_col_size, int *kernel,
		int output_row_size, int output_col_size, ull *output
		)
{
	int total_cells = output_row_size * output_col_size;
        int start_cell = (blockIdx.x*blockDim.x + threadIdx.x);
        if(start_cell >= total_cells) return;
        int output_row = start_cell / output_col_size;
	int output_col = start_cell % output_col_size;
        for(int kernel_row = 0; kernel_row< kernel_row_size; kernel_row++)
	{
		for(int kernel_col = 0; kernel_col< kernel_col_size; kernel_col++)
		{
			int input_row = (output_row + 2*kernel_row) % input_row_size;
			int input_col = (output_col + 2*kernel_col) % input_col_size;
			output[output_row * output_col_size + output_col] += input[input_row * input_col_size +input_col] 
				* kernel[kernel_row * kernel_col_size +kernel_col];
		}
	}
}

// Fill in this function
void gpuThread(int input_row_size, int input_col_size, int *input, 
                int kernel_row_size, int kernel_col_size, int *kernel,
                int output_row_size, int output_col_size, ull *output) 
{
    cudaFree(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  // Assumes you have a single GPU device

	//copy data to device
	int *dev_input, *dev_kernel;
	ull *dev_output;
	cudaMalloc((void**)&dev_input, sizeof(int) * input_row_size * input_col_size);
	cudaMalloc((void**)&dev_kernel, sizeof(int) * kernel_row_size * kernel_col_size);
	cudaMalloc((void**)&dev_output, sizeof(ull) * output_row_size * output_col_size);
	cudaMemcpy(dev_input, input, sizeof(int) * input_row_size*input_col_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kernel, kernel, sizeof(int) * kernel_row_size*kernel_col_size, cudaMemcpyHostToDevice);
	// Kernel invocation

        int maxThreadsNeeded = output_row_size * output_col_size;
        int blocksWeCreate = ceil((double)maxThreadsNeeded / deviceProp.maxThreadsPerBlock);
        dim3 threadsPerBlock(deviceProp.maxThreadsPerBlock);
	dim3 numBlocks(blocksWeCreate); //N / threadsPerBlock.x, N / threadsPerBlock.y)
	Convolute<<<numBlocks, threadsPerBlock>>>(input_row_size, input_col_size, dev_input, kernel_row_size, kernel_col_size, dev_kernel, output_row_size, output_col_size, dev_output);
	cudaDeviceSynchronize();
	cudaMemcpy(output, dev_output, sizeof(ull) * output_row_size*output_col_size, cudaMemcpyDeviceToHost);
}

