#pragma once
#pragma GCC diagnostic ignored "-Wregister"

#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>
#include<math.h>

#include "utility.h"

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
			int input_row = output_row + kernel_row;
			int input_col = output_col + kernel_col;
			output[start_cell] += input[input_row * input_col_size +input_col] 
				* kernel[kernel_row * kernel_col_size +kernel_col];
			//if(output_row==1 && output_col==0) {
			//	printf("input[%d][%d]: %d\t", input_row, input_col, input[input_row * input_col_size + input_col]);
			//	printf("output[%d][%d]: %llu\t", output_row, output_col, output[output_row * output_col_size + output_col]);
			//	printf("kernel[%d][%d]: %d\t", kernel_row, kernel_col, kernel[kernel_row * kernel_col_size + kernel_col]);
			//	printf("\n");
                        //}
		}
	}
}

// Fill in this function
void gpuThread(int input_row_size, int input_col_size, int *input, 
                int kernel_row_size, int kernel_col_size, int *kernel,
                int output_row_size, int output_col_size, ull *output) 
{
        int output_row_half_addr, output_col_half_addr;
        int partial_output_row_half_addr, partial_output_col_half_addr;
        int row_offset, col_offset;


        //create four sub-blocks of input with padding
        int padded_input_row_size = input_row_size + 1*kernel_row_size;
        int padded_input_col_size = input_col_size + 1*kernel_col_size;
        int quarter_input_row_sizes[4] = {(padded_input_row_size+1)/2, (padded_input_row_size+1)/2, (padded_input_row_size)/2, (padded_input_row_size)/2};
        int quarter_input_col_sizes[4] = {(padded_input_col_size+1)/2, (padded_input_col_size)/2, (padded_input_col_size+1)/2, (padded_input_col_size)/2};

        int quarter_output_row_sizes[4] = {(output_row_size+1)/2, (output_row_size+1)/2, (output_row_size)/2, (output_row_size)/2};
        int quarter_output_col_sizes[4] = {(output_col_size+1)/2, (output_col_size)/2, (output_col_size+1)/2, (output_col_size)/2};
        
        int *quarter_inputs[4];
        
        int *dev_input[4], *dev_kernel;
        ull *dev_output[4];
    cudaFree(0);
    cudaDeviceProp deviceProp;
            cudaMalloc((void**)&dev_kernel, sizeof(int) * kernel_row_size * kernel_col_size);
            cudaMemcpyAsync(dev_kernel, kernel, sizeof(int) * kernel_row_size*kernel_col_size, cudaMemcpyHostToDevice);
        // create four sub-blocks of output
        for(int i=0; i<4; ++i) {
            //copy data to device
            cudaMalloc((void**)&dev_input[i], sizeof(int) * input_row_size * input_col_size);
            cudaMalloc((void**)&dev_output[i], sizeof(ull) * output_row_size * output_col_size);
                quarter_inputs[i] = new int[quarter_input_row_sizes[i] * quarter_input_col_sizes[i]]; // new array to store each of the input matrix with padding.
                row_offset = i>>1;
                col_offset = i&1;
                createQuarterArray(input_row_size, input_col_size, input,
                                quarter_input_row_sizes[i], quarter_input_col_sizes[i], quarter_inputs[i]
                                , row_offset, col_offset); // fill one of the four subblocks of the input matrix with padding
            cudaMemcpyAsync(dev_input[i], quarter_inputs[i], sizeof(int) * quarter_input_row_sizes[i]*quarter_input_col_sizes[i], cudaMemcpyHostToDevice);
            int maxThreadsNeeded = quarter_output_row_sizes[i] * quarter_output_col_sizes[i];
            int threadsWeCreate = deviceProp.maxThreadsPerBlock;
            int blocksWeCreate = ceil((double)maxThreadsNeeded / threadsWeCreate);
            dim3 threadsPerBlock(threadsWeCreate);
            dim3 numBlocks(blocksWeCreate);
            //std::cout << "starting kernel\n";
            Convolute<<<numBlocks, threadsPerBlock>>>(quarter_input_row_sizes[i], quarter_input_col_sizes[i], dev_input[i], kernel_row_size, kernel_col_size, dev_kernel, quarter_output_row_sizes[i], quarter_output_col_sizes[i], dev_output[i]);
                delete [] quarter_inputs[i];
                cudaFree(dev_input[i]);
        }


        // merge back the four output blocks
        for(int i=0; i<4; ++i) {
            ull *quarter_output = new ull[quarter_output_row_sizes[i] * quarter_output_col_sizes[i]]; // new array to st
            cudaMemcpy(quarter_output, dev_output[i], sizeof(ull) * quarter_output_row_sizes[i]*quarter_output_col_sizes[i], cudaMemcpyDeviceToHost);
            cudaFree(dev_output[i]);
                row_offset = i>>1;
                col_offset = i&1;
                output_row_half_addr = row_offset * output_col_size;
                partial_output_row_half_addr = 0;
                for(int row=0; row<quarter_output_row_sizes[i]; ++row) {
                        output_col_half_addr = output_row_half_addr + col_offset;
                        partial_output_col_half_addr = partial_output_row_half_addr;
                        for(int col=0; col<quarter_output_col_sizes[i]; ++col) {
                                output[output_col_half_addr] = quarter_output[partial_output_col_half_addr];
                                output_col_half_addr += 2;
                                partial_output_col_half_addr += 1;
                        }
                        output_row_half_addr += output_col_size<<1;
                        partial_output_row_half_addr += quarter_output_col_sizes[i];
                }
                delete [] quarter_output;
        }
}
