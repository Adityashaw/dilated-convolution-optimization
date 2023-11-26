#pragma once
#pragma GCC diagnostic ignored "-Wregister"

#include <iostream>

#define ull unsigned long long int

void createQuarterArray(register int input_row_size, register int input_col_size, register int *input,
                        register int output_row_size, register int output_col_size, register int *output,
                        register int row_offset, register int col_offset)
{
    // Padding the input array with repetitions of the first (padded_input_col_size - input_col_size) columns at the end
    // Now padding the array with repetitions of the first (padded_input_row_size - input_row) rows at the bottom

    register int row, col, temp_col;
    register int input_row_half_addr, output_row_half_addr;
    register int limit_row = (input_row_size - 1 - row_offset) / 2;
    register int limit_col = (input_col_size - 1 - col_offset) / 2;
    input_row_half_addr = row_offset * input_col_size;
    output_row_half_addr = 0;
    for (row = 0; row <= limit_row; ++row)
    {
        col = 0;
        temp_col = col_offset;
        for (; col <= limit_col - 4; col += 4, temp_col += 8)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
            output[output_row_half_addr + col + 1] = input[input_row_half_addr + temp_col + 2];
            output[output_row_half_addr + col + 2] = input[input_row_half_addr + temp_col + 4];
            output[output_row_half_addr + col + 3] = input[input_row_half_addr + temp_col + 6];
        }
        for (; col <= limit_col; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
        for (temp_col = temp_col - input_col_size; col < output_col_size; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
    }

    input_row_half_addr -= input_row_size * input_col_size;
    for (; row < output_row_size; ++row)
    {
        for (col = 0, temp_col = col_offset; col <= limit_col; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
        for (temp_col = temp_col - input_col_size; col < output_col_size; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = input[input_row_half_addr + temp_col];
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
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
