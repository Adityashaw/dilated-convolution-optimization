#pragma once
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#include <immintrin.h>

#define ull unsigned long long int

int multiplyAndReduce16bit16Integers(const short int* , const short int* , int, int&, int, int);
int multiplyAndReduce16bit8Integers(const short int* , const short int* , int, int&, int, int);
void printArray(int, int, ull *);
void printArrayInt(int, int, int *);
void printArrayShortInt(int, int, short int *);

void convolute(int input_row_size, int input_col_size, short int *input, int kernel_row_size, int kernel_col_size,
               short int *kernel, int output_row_start, int output_row_end, int output_col_size, ull *output)
{
    int input_row, input_col;
    int output_row, output_col;
    int kernel_row, kernel_col;
    int kernel_row_half_addr;
    int input_row_half_addr, output_row_half_addr;
    int kernel_col_size_padded_16 = kernel_col_size - kernel_col_size % 16;
    int kernel_col_size_padded_8 = kernel_col_size - kernel_col_size % 8;
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
                for (kernel_col = 0; kernel_col < kernel_col_size_padded_16; kernel_col += 16)
                {
                    temp_output += multiplyAndReduce16bit16Integers(input, kernel, input_row_half_addr, input_col, kernel_row_half_addr, kernel_col);
                }
                for (; kernel_col < kernel_col_size_padded_8; kernel_col += 8)
                {
                    temp_output += multiplyAndReduce16bit8Integers(input, kernel, input_row_half_addr, input_col, kernel_row_half_addr, kernel_col);
                }
                for (; kernel_col < kernel_col_size; kernel_col += 1)
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
                        short int *output, int row_offset, int col_offset)
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

int getNumCores()
{
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#elif MACOS
    int nm[2];
    size_t len = 4;
    uint32_t count;

    nm[0] = CTL_HW;
    nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);

    if (count < 1)
    {
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        if (count < 1)
        {
            count = 1;
        }
    }
    return count;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

int multiplyAndReduce16bit16Integers(const short int* input, const short int* kernel, int input_row_half_addr, int& input_col, int kernel_row_half_addr, int kernel_col) {
                    __m256i vector1 = _mm256_loadu_si256((__m256i *)&input[input_row_half_addr + input_col]);
                    __m256i vector2 = _mm256_loadu_si256((__m256i *)&kernel[kernel_row_half_addr + kernel_col]);
                    __m256i result = _mm256_mullo_epi16(vector1, vector2);

                    // Convert the 16-bit 16 integers to 32-bit 16 integers
                    __m256i low128_256 =
                        _mm256_cvtepu16_epi32((__m128i)_mm256_castsi256_si128(result)); // Extract lower 128 bits (8 integers)
                    __m256i high128_256 = _mm256_cvtepu16_epi32(
                        (__m128i)_mm256_extracti128_si256(result, 1)); // Extract higher 128 bits (8 integers)
                    __m256i accumulated = _mm256_add_epi32(low128_256, high128_256); // Accumulate the 32-bit 8 integers

                    __m128i low128 = _mm256_castsi256_si128(accumulated);       // Extract lower 128 bits (4 integers)
                    __m128i high128 = _mm256_extracti128_si256(accumulated, 1); // Extract higher 128 bits (4 integers)
                    __m128i sum128 = _mm_add_epi32(low128, high128);            // Add the two 128-bit results
                    sum128 = _mm_hadd_epi32(sum128, sum128);                    // Perform horizontal addition

                    // Access the result
                    int32_t *resultArray = (int32_t *)&sum128;

                    // Extract the first two 32-bit integers and add them
                    int32_t sum = resultArray[0] + resultArray[1];

                    input_col += 16;
                    return sum;
                    // try to replace with printAt() function
                    // if(input_row==0 && input_col==0)
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

int multiplyAndReduce16bit8Integers(const short int* input, const short int* kernel, int input_row_half_addr, int& input_col, int kernel_row_half_addr, int kernel_col) {
                    __m128i vector1 = _mm_loadu_si128((__m128i *)&input[input_row_half_addr + input_col]);
                    __m128i vector2 = _mm_loadu_si128((__m128i *)&kernel[kernel_row_half_addr + kernel_col]);
                    __m128i result = _mm_mullo_epi16(vector1, vector2);

                    // Convert the 16-bit integers to 32-bit integers
                    __m128i lowPart = _mm_cvtepu16_epi32(_mm_unpacklo_epi64(result, result));
                    __m128i highPart = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(result, result));

                    // Add the extended values
                    __m128i sum128 = _mm_add_epi32(lowPart, highPart);

                    sum128 = _mm_hadd_epi32(sum128, sum128); // Perform horizontal addition

                    // Access the result
                    int32_t *resultArray = (int32_t *)&sum128;

                    // Extract the first two 32-bit integers and add them
                    int32_t sum = resultArray[0] + resultArray[1];

                    // output[output_row_half_addr + output_col] += sum;
                    input_col += 8;
                    return sum;
                    // try to replace with printAt() function
                    // if(input_row==0 && input_col==0)
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
