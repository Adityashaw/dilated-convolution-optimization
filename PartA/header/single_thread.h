#include<algorithm>
#include <immintrin.h>
#include <iostream>
#include "utility.h"

#define ull unsigned long long int

// Optimize this function
void singleThread( int input_row_size, int input_col_size, int *input, 
                int kernel_row_size, int kernel_col_size, int *kernel,
                int output_row_size, int output_col_size, ull *output) 
{
        

        int output_row_half_addr, output_col_half_addr;
        int partial_output_row_half_addr, partial_output_col_half_addr;
        int row_offset, col_offset;

        short int* kernel2 = new short int[kernel_row_size * kernel_col_size];
        for(int i=0; i<kernel_row_size; ++i) {
                for(int j=0; j<kernel_col_size; ++j) {
                        kernel2[i*kernel_col_size + j] = (short int)kernel[i*kernel_col_size + j];
                }
        }

        //create four sub-blocks of input with padding
        int padded_input_row_size = input_row_size + 1*kernel_row_size;
        int padded_input_col_size = input_col_size + 1*kernel_col_size;
        int quarter_input_row_sizes[4] = {(padded_input_row_size+1)/2, (padded_input_row_size+1)/2, (padded_input_row_size)/2, (padded_input_row_size)/2};
        int quarter_input_col_sizes[4] = {(padded_input_col_size+1)/2, (padded_input_col_size)/2, (padded_input_col_size+1)/2, (padded_input_col_size)/2};

        int quarter_output_row_sizes[4] = {(output_row_size+1)/2, (output_row_size+1)/2, (output_row_size)/2, (output_row_size)/2};
        int quarter_output_col_sizes[4] = {(output_col_size+1)/2, (output_col_size)/2, (output_col_size+1)/2, (output_col_size)/2};
        
        short int *quarter_inputs[4];
        ull *quarter_outputs[4];
        
        // create four sub-blocks of output
        for(int i=0; i<4; ++i) {
                quarter_inputs[i] = new short int[quarter_input_row_sizes[i] * quarter_input_col_sizes[i]]; // new array to store each of the input matrix with padding.
                row_offset = i>>1;
                col_offset = i&1;
                createQuarterArray(input_row_size, input_col_size, input,
                                quarter_input_row_sizes[i], quarter_input_col_sizes[i], quarter_inputs[i]
                                , row_offset, col_offset); // fill one of the four subblocks of the input matrix with padding
                quarter_outputs[i] = new ull[quarter_output_row_sizes[i] * quarter_output_col_sizes[i]](); // new array to st
                convolute(quarter_input_row_sizes[i], quarter_input_col_sizes[i], quarter_inputs[i],
                                kernel_row_size, kernel_col_size, kernel2,
                                0, quarter_output_row_sizes[i], quarter_output_col_sizes[i], quarter_outputs[i]); //basic convolution
                //std::cout << "input\n";
                //printArrayShortInt(quarter_input_row_sizes[i], quarter_input_col_sizes[i], quarter_inputs[i]);
                //std::cout << "output\n";
                //printArray(quarter_output_row_sizes[i], quarter_output_col_sizes[i], quarter_outputs[i]);
                delete [] quarter_inputs[i];
        }
        delete [] kernel2;


        // merge back the four output blocks
        for(int i=0; i<4; ++i) {
                row_offset = i>>1;
                col_offset = i&1;
                output_row_half_addr = row_offset * output_col_size;
                partial_output_row_half_addr = 0;
                for(int row=0; row<quarter_output_row_sizes[i]; ++row) {
                        output_col_half_addr = output_row_half_addr + col_offset;
                        partial_output_col_half_addr = partial_output_row_half_addr;
                        for(int col=0; col<quarter_output_col_sizes[i]; ++col) {
                                output[output_col_half_addr] = quarter_outputs[i][partial_output_col_half_addr];
                                output_col_half_addr += 2;
                                partial_output_col_half_addr += 1;
                        }
                        output_row_half_addr += output_col_size<<1;
                        partial_output_row_half_addr += quarter_output_col_sizes[i];
                }
                delete [] quarter_outputs[i];
        }
        //printArray(output_row_size, output_col_size, output);
        return;
}
