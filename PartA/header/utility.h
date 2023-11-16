#pragma once
#include <iostream>

#define ull unsigned long long int

void printArray(int, int, ull *);
void printArrayInt(int, int, int *);
void printArrayShortInt(int, int, short int *);

void convolute(int input_row_size, int input_col_size, int *input, int kernel_row_size, int kernel_col_size,
               int *kernel, int output_row_start, int output_row_end, int output_col_size, ull *output)
{
    int input_col;
    int output_row, output_col;
    int kernel_row, kernel_col;
    int kernel_row_half_addr;
    int input_row_half_addr, output_row_half_addr;
    register ull temp_output;

    output_row_half_addr = output_row_start * output_col_size;
    for (output_row = output_row_start; output_row < output_row_end; ++output_row)
    {
        // try to use dynamic programming by reusing previous state
        // instead of multiplying everytime.

        for (output_col = 0; output_col < output_col_size; output_col += 1)
        {
            input_row_half_addr = output_row * input_col_size;
            kernel_row_half_addr = 0;
            temp_output = 0;
            for (kernel_row = 0; kernel_row < kernel_row_size; kernel_row += 1)
            {
                input_col = output_col;
                for (kernel_col = 0; kernel_col < kernel_col_size; kernel_col += 1)
                {
                    temp_output += input[input_row_half_addr + input_col] * kernel[kernel_row_half_addr + kernel_col];
                    ++input_col;
                    // try to replace with printAt() function
                    // if(input_row==0 && input_col==0) {
                    // if (output_row == 0 && output_col == 0)
                    //{
                    //    std::cout << "input[" << input_row << "][" << input_col << "]:"
                    //              << "\t"; // input[input_row_half_addr + input_col] << "\t";
                    //    std::cout << "output[" << output_row << "][" << output_col << "]:"
                    //              << "\t"; // output[output_row_half_addr + output_col] << "\t";
                    //    std::cout << "kernel[" << kernel_row << "][" << kernel_col << "]\t";
                    //    std::cout << std::endl;
                    //}
                }
                input_row_half_addr += input_col_size;
                kernel_row_half_addr += kernel_col_size;
            }
            output[output_row_half_addr + output_col] = temp_output;
        }
        output_row_half_addr += output_col_size;
    }
}

void createQuarterArray(int input_row_size, int input_col_size, int *input, int output_row_size, int output_col_size,
                        int *output, int row_offset, int col_offset)
{
    // Padding the input array with repetitions of the first (padded_input_col_size - input_col_size) columns at the end
    // Now padding the array with repetitions of the first (padded_input_row_size - input_row) rows at the bottom

    int row, col;
    int input_row_half_addr, output_row_half_addr;
    int limit_row = (input_row_size - 1 - row_offset) / 2;
    int limit_col = (input_col_size - 1 - col_offset) / 2;
    for (row = 0; row <= limit_row; ++row)
    {
        input_row_half_addr = (2 * row + row_offset) * input_col_size;
        output_row_half_addr = row * output_col_size;
        int temp_col;
        for (col = 0, temp_col = col_offset; col <= limit_col; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
        for (temp_col = temp_col - input_col_size; col < output_col_size; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
    }

    for (; row < output_row_size; ++row)
    {
        input_row_half_addr = (2 * row + row_offset - input_row_size) * input_col_size;
        output_row_half_addr = row * output_col_size;
        int temp_col;
        for (col = 0, temp_col = col_offset; col <= limit_col; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
        for (temp_col = temp_col - input_col_size; col < output_col_size; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
    }
}

void printArray(int array_row_size, int array_col_size, ull *array)
{
    for (int row = 0; row < array_row_size; ++row)
    {
        int row_half_addr = row * array_col_size;
        for (int col = 0; col < array_col_size; ++col)
        {
            std::cout << array[row_half_addr + col] << " ";
        }
        std::cout << std::endl;
    }
}

void printArrayInt(int array_row_size, int array_col_size, int *array)
{
    for (int row = 0; row < array_row_size; ++row)
    {
        int row_half_addr = row * array_col_size;
        for (int col = 0; col < array_col_size; ++col)
        {
            std::cout << array[row_half_addr + col] << " ";
        }
        std::cout << std::endl;
    }
}

void printArrayShortInt(int array_row_size, int array_col_size, short int *array)
{
    for (int row = 0; row < array_row_size; ++row)
    {
        int row_half_addr = row * array_col_size;
        for (int col = 0; col < array_col_size; ++col)
        {
            std::cout << array[row_half_addr + col] << " ";
        }
        std::cout << std::endl;
    }
}

void printAt(int input_row_size, int input_col_size, int *input, int kernel_row_size, int kernel_col_size, int *kernel,
             int output_row_size, int output_col_size, long long unsigned int *output)
{
    // how to use this function???
    /*int input_row, input_col;
    int output_row, output_col;
    int kernel_row, kernel_col;
    //if(input_row==0 && input_col==0) {
    if(output_row==0 && output_col==0) {
            std::cout << "output[" << output_row << "][" << output_col << "]" << std::endl;
            std::cout << "kernel[" << kernel_row << "][" << kernel_col << "]" << std::endl;
    }
    */
}
