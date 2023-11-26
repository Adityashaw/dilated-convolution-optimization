#pragma once
#pragma GCC diagnostic ignored "-Wregister"
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
#define force_inline __attribute__((always_inline)) inline

static int multiplyAndReduce16bit16Integers(const short int *, const short int *, int, int, int, int);
static void multiplyAndReduce16bit16IntegersTwoRows(const short int *, const short int *, int, int, int, int, int, ull&, ull&);
static int multiplyAndReduce16bit8Integers(const short int *, const short int *, int, int, int, int);
static void multiplyAndReduce16bit8IntegersTwoRows(const short int *, const short int *, int, int, int, int, int, ull&, ull&);
static int multiplyAndReduce16bit4Integers(const short int *, const short int *, int, int, int, int);
static void multiplyAndReduce16bit4IntegersTwoRows(const short int *, const short int *, int, int, int, int, int, ull&, ull&);
void printArray(int, int, ull *);
void printArrayInt(int, int, int *);
void printArrayShortInt(int, int, short int *);

force_inline void convolute(register int input_row_size, register int input_col_size, register short int *input,
               register int kernel_row_size, register int kernel_col_size, register short int *kernel,
               register int output_row_start, register int output_row_end, register int output_col_size,
               register ull *output)
{
    register int input_col;
    register int output_row, output_col;
    register int kernel_row, kernel_col;
    register int kernel_row_half_addr;
    register int input_row_half_addr, output_row_half_addr;
    register int input_row_half_addr2, output_row_half_addr2;
    // int kernel_col_size_padded_32 = kernel_col_size - kernel_col_size % 32;
    register int kernel_col_size_padded_16 = kernel_col_size - kernel_col_size % 16;
    register int kernel_col_size_padded_8 = kernel_col_size - kernel_col_size % 8;
    register int kernel_col_size_padded_4 = kernel_col_size - kernel_col_size % 4;
    register int output_row_size_padded_2 = output_row_end - output_row_end % 2;
    register ull temp_output, temp_output2;

    output_row_half_addr = output_row_start * output_col_size;
    output_row = output_row_start;
    output_row_half_addr2 = output_row_half_addr + output_col_size;
    for (; output_row < output_row_size_padded_2; output_row+=2)
    {
        // try to use dynamic programming by reusing previous state
        // instead of multiplying everytime.

        output_col = 0;
        for (; output_col < output_col_size; output_col += 1)
        {
            input_row_half_addr = output_row * input_col_size;
            input_row_half_addr2 = input_row_half_addr + input_col_size;
            kernel_row_half_addr = 0;
            temp_output = 0;
            temp_output2 = 0;
            for (kernel_row = 0; kernel_row < kernel_row_size; kernel_row += 1)
            {
                input_col = output_col;
                for (kernel_col = 0; kernel_col < kernel_col_size_padded_16; kernel_col += 16)
                {
                    multiplyAndReduce16bit16IntegersTwoRows(input, kernel, input_row_half_addr, input_row_half_addr2, input_col,
                                                                    kernel_row_half_addr, kernel_col, temp_output, temp_output2);
                    input_col += 16;
                }
                for (; kernel_col < kernel_col_size_padded_8; kernel_col += 8)
                {
                    multiplyAndReduce16bit8IntegersTwoRows(input, kernel, input_row_half_addr, input_row_half_addr2, input_col,
                                                                    kernel_row_half_addr, kernel_col, temp_output, temp_output2);
                    input_col += 8;
                }
                for (; kernel_col < kernel_col_size_padded_4; kernel_col += 4)
                {
                    multiplyAndReduce16bit4IntegersTwoRows(input, kernel, input_row_half_addr, input_row_half_addr2, input_col,
                                                                    kernel_row_half_addr, kernel_col, temp_output, temp_output2);
                    input_col += 4;
                }
                for (; kernel_col < kernel_col_size; kernel_col += 1)
                {
                    temp_output += input[input_row_half_addr + input_col] * kernel[kernel_row_half_addr + kernel_col];
                    temp_output2 += input[input_row_half_addr2 + input_col] * kernel[kernel_row_half_addr + kernel_col];
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
                input_row_half_addr2 += input_col_size;
                kernel_row_half_addr += kernel_col_size;
            }
            output[output_row_half_addr + output_col] = temp_output;
            output[output_row_half_addr2 + output_col] = temp_output2;
        }
        output_row_half_addr += 2*output_col_size;
        output_row_half_addr2 += 2*output_col_size;
    }
    for (; output_row < output_row_end; ++output_row)
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
                    temp_output += multiplyAndReduce16bit16Integers(input, kernel, input_row_half_addr, input_col,
                                                                    kernel_row_half_addr, kernel_col);
                    input_col += 16;
                }
                for (; kernel_col < kernel_col_size_padded_8; kernel_col += 8)
                {
                    temp_output += multiplyAndReduce16bit8Integers(input, kernel, input_row_half_addr, input_col,
                                                                   kernel_row_half_addr, kernel_col);
                    input_col += 8;
                }
                for (; kernel_col < kernel_col_size_padded_4; kernel_col += 4)
                {
                    temp_output += multiplyAndReduce16bit4Integers(input, kernel, input_row_half_addr, input_col,
                                                                   kernel_row_half_addr, kernel_col);
                    input_col += 4;
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

static force_inline void copy16IntTo16ShortIntTwiceUsingTwoIndices(int array_idx, int array_idx2, int *array,
                                                                   short int *array2)
{
    // Load the input data into a 256-bit AVX2 register
    __m256i input = _mm256_loadu_si256((__m256i *)(array + array_idx));
    __m256i input_b = _mm256_loadu_si256((__m256i *)(array + array_idx + 8));
    __m256i input2 = _mm256_loadu_si256((__m256i *)(array + array_idx2));
    __m256i input2_b = _mm256_loadu_si256((__m256i *)(array + array_idx2 + 8));

    // Convert the 32-bit integers to 16-bit integers using AVX2
    __m256i result = _mm256_packs_epi32(input, input_b);
    __m256i result2 = _mm256_packs_epi32(input2, input2_b);

    // Store the result
    _mm256_storeu_si256((__m256i *)(array2 + array_idx), _mm256_permute4x64_epi64(result, 0xD8));
    _mm256_storeu_si256((__m256i *)(array2 + array_idx2), _mm256_permute4x64_epi64(result2, 0xD8));
}

static force_inline void copy16IntTo16ShortInt(int array_idx, int *array, short int *array2)
{
    // Load the input data into a 256-bit AVX2 register
    __m256i input = _mm256_loadu_si256((__m256i *)(array + array_idx));
    __m256i input_b = _mm256_loadu_si256((__m256i *)(array + array_idx + 8));

    // Convert the 32-bit integers to 16-bit integers using AVX2
    __m256i result = _mm256_packs_epi32(input, input_b);

    // Store the result
    _mm256_storeu_si256((__m256i *)(array2 + array_idx), _mm256_permute4x64_epi64(result, 0xD8));
}

static force_inline void copy8IntTo8ShortIntWith1Dilation(int array_idx, int *array, int array2_idx, short int *array2)
{
    // Load the input data into a 256-bit AVX2 register
    __m256i input = _mm256_loadu_si256((__m256i *)(array + array_idx));
    __m256i permute_vector = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0); //0x02461357);
    __m256i input_b = _mm256_loadu_si256((__m256i *)(array + array_idx + 8));

    // Permute to get elements at correct indices
    input = _mm256_permutevar8x32_epi32(input, permute_vector);
    input_b = _mm256_permutevar8x32_epi32(input_b, permute_vector);

    // Convert the 32-bit integers to 16-bit integers using AVX2
    __m256i result = _mm256_packus_epi32(input, input_b);

    // Store the result
    _mm_storeu_si128((__m128i *)(array2 + array2_idx), _mm256_castsi256_si128(result)); //_mm256_permute4x64_epi64(result, 0xD8));
}

static force_inline void copy8IntTo8ShortIntWith1DilationTo2Arrays(int array_idx, int *array, int array2_idx, short int *array2, short int *array3)
{
    // Load the input data into a 256-bit AVX2 register
    __m256i input = _mm256_loadu_si256((__m256i *)(array + array_idx));
    __m256i permute_vector = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0); //0x02461357);
    __m256i input_b = _mm256_loadu_si256((__m256i *)(array + array_idx + 8));

    // Permute to get elements at correct indices
    input = _mm256_permutevar8x32_epi32(input, permute_vector);
    input_b = _mm256_permutevar8x32_epi32(input_b, permute_vector);

    // Convert the 32-bit integers to 16-bit integers using AVX2
    __m256i result = _mm256_packus_epi32(input, input_b);

    // Store the result
    _mm_storeu_si128((__m128i *)(array2 + array2_idx), _mm256_castsi256_si128(result)); //_mm256_permute4x64_epi64(result, 0xD8));
    _mm_storeu_si128((__m128i *)(array3 + array2_idx), _mm256_extracti128_si256(result, 1)); //_mm256_permute4x64_epi64(result, 0xD8));
}

static force_inline void copy8IntTo8ShortInt(int array_idx, int *array, short int *array2)
{
    // Load the input data into a 128-bit AVX2 register
    __m256i input = _mm256_loadu_si256((__m256i *)(array + array_idx));

    // Convert the 32-bit integers to 16-bit integers using AVX2
    input = _mm256_packs_epi32(input, _mm256_setzero_si256());

    // Store the result
    _mm_storeu_si128((__m128i *)(array2 + array_idx), _mm256_castsi256_si128(_mm256_permute4x64_epi64(input, 0xD8)));
}

static force_inline void copy8IntTo8ShortIntTwiceUsingTwoIndices(int array_idx, int array_idx2, int *array,
                                                                   short int *array2)
{
            // Load the input data into a 256-bit AVX2 register
    __m256i input = _mm256_loadu_si256((__m256i *)(array + array_idx));
    __m256i input2 = _mm256_loadu_si256((__m256i *)(array + array_idx2));

            // Convert the 32-bit integers to 16-bit integers using AVX2

            __m256i result = _mm256_packs_epi32(input, _mm256_setzero_si256());
            __m256i result2 = _mm256_packs_epi32(input2, _mm256_setzero_si256());

            // Store the result
    _mm_storeu_si128((__m128i *)(array2 + array_idx), _mm256_castsi256_si128(_mm256_permute4x64_epi64(result, 0xD8)));
    _mm_storeu_si128((__m128i *)(array2 + array_idx2), _mm256_castsi256_si128(_mm256_permute4x64_epi64(result2, 0xD8)));
}

force_inline void copyKernel(int kernel_col_size, int kernel_row_size, int *kernel, short int *kernel2)
{
    register int kernel_col_size_padded_16 = kernel_col_size - kernel_col_size % 16;
    register int kernel_col_size_padded_8 = kernel_col_size - kernel_col_size % 8;
    register int kernel_col_size_padded_4 = kernel_col_size - kernel_col_size % 4;
    register int kernel_row_size_padded_2 = kernel_row_size - kernel_row_size % 2;
    register int kernel_idx = 0, kernel_idx2 = kernel_col_size;
    register int row, col;
    row = 0;
    for (; row < kernel_row_size_padded_2; row += 2)
    {
        for (col = 0; col < kernel_col_size_padded_16; col += 16)
        {
            copy16IntTo16ShortIntTwiceUsingTwoIndices(kernel_idx, kernel_idx2, kernel, kernel2);
            kernel_idx += 16;
            kernel_idx2 += 16;
        }
        for (; col < kernel_col_size_padded_8; col += 8)
        {
            copy8IntTo8ShortIntTwiceUsingTwoIndices(kernel_idx, kernel_idx2, kernel, kernel2);
            kernel_idx += 8;
            kernel_idx2 += 8;
        }
        if (col < kernel_col_size_padded_4)
        {
            kernel2[kernel_idx] = (short int)kernel[kernel_idx];
            kernel2[kernel_idx + 1] = (short int)kernel[kernel_idx + 1];
            kernel2[kernel_idx + 2] = (short int)kernel[kernel_idx + 2];
            kernel2[kernel_idx + 3] = (short int)kernel[kernel_idx + 3];
            kernel_idx += 4;
            kernel2[kernel_idx2] = (short int)kernel[kernel_idx2];
            kernel2[kernel_idx2 + 1] = (short int)kernel[kernel_idx2 + 1];
            kernel2[kernel_idx2 + 2] = (short int)kernel[kernel_idx2 + 2];
            kernel2[kernel_idx2 + 3] = (short int)kernel[kernel_idx2 + 3];
            kernel_idx2 += 4;
            col += 4;
        }
        for (; col < kernel_col_size; col += 1)
        {
            kernel2[kernel_idx] = (short int)kernel[kernel_idx];
            kernel_idx += 1;
            kernel2[kernel_idx2] = (short int)kernel[kernel_idx2];
            kernel_idx2 += 1;
        }
        // std::cout << kernel_idx << "\t" << kernel_idx2 << std::endl;
        kernel_idx += kernel_col_size;
        kernel_idx2 += kernel_col_size;
    }
    for (; row < kernel_row_size; ++row)
    {
        for (col = 0; col < kernel_col_size_padded_16; col += 16)
        {
            copy16IntTo16ShortInt(kernel_idx, kernel, kernel2);
            kernel_idx += 16;
        }
        for (; col < kernel_col_size_padded_8; col += 8)
        {
            copy8IntTo8ShortInt(kernel_idx, kernel, kernel2);
            kernel_idx += 8;
        }
        if (col < kernel_col_size_padded_4)
        {
            kernel2[kernel_idx] = (short int)kernel[kernel_idx];
            kernel2[kernel_idx + 1] = (short int)kernel[kernel_idx + 1];
            kernel2[kernel_idx + 2] = (short int)kernel[kernel_idx + 2];
            kernel2[kernel_idx + 3] = (short int)kernel[kernel_idx + 3];
            kernel_idx += 4;
            col += 4;
        }
        for (; col < kernel_col_size; col += 1)
        {
            kernel2[kernel_idx] = (short int)kernel[kernel_idx];
            kernel_idx += 1;
        }
    }
}

force_inline void createQuarterArray(register int input_row_size, register int input_col_size, register int *input,
                        register int output_row_size, register int output_col_size, register short int *output,
                        register int row_offset, register int col_offset)
{
    // Padding the input array with repetitions of the first (padded_input_col_size - input_col_size) columns at the end
    // Now padding the array with repetitions of the first (padded_input_row_size - input_row) rows at the bottom

    //std::cout << "output size:[" << output_row_size << "][" << output_col_size << std::endl;
    register int row, col, temp_col;
    register int input_row_half_addr, output_row_half_addr;
    register int limit_row = (input_row_size - 1 - row_offset) / 2;
    register int limit_col = (input_col_size - 1 - col_offset) / 2;
    input_row_half_addr = row_offset * input_col_size;
    output_row_half_addr = 0;
    for (row = 0; row <= limit_row; ++row)
    {
        col = 0; temp_col = col_offset;
        for(; col <= limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1Dilation(input_row_half_addr + temp_col, input, output_row_half_addr + col, output);
        }
        for (; col <= limit_col - 4; col += 4, temp_col += 8)
        {
            output[output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            output[output_row_half_addr + col + 1] = (short int)input[input_row_half_addr + temp_col + 2];
            output[output_row_half_addr + col + 2] = (short int)input[input_row_half_addr + temp_col + 4];
            output[output_row_half_addr + col + 3] = (short int)input[input_row_half_addr + temp_col + 6];
        }
        for (; col <= limit_col; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
        }
        for (temp_col = temp_col - input_col_size; col < output_col_size; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
    }

    input_row_half_addr -= input_row_size * input_col_size;
    for (; row < output_row_size; ++row)
    {
        col = 0, temp_col = col_offset;
        for(; col <= limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1Dilation(input_row_half_addr + temp_col, input, output_row_half_addr + col, output);
        }
        for (; col <= limit_col - 4; col += 4, temp_col += 8)
        {
            output[output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            output[output_row_half_addr + col + 1] = (short int)input[input_row_half_addr + temp_col + 2];
            output[output_row_half_addr + col + 2] = (short int)input[input_row_half_addr + temp_col + 4];
            output[output_row_half_addr + col + 3] = (short int)input[input_row_half_addr + temp_col + 6];
        }
        for (; col <= limit_col; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
        }
        for (temp_col = temp_col - input_col_size; col < output_col_size; ++col, temp_col += 2)
        {
            output[output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
    }
}

void createQuarterArraysWithEvenEvenInput(register int input_row_size, register int input_col_size, register int *input,
                        register int output_row_size, register int output_col_size, register short int *outputs[4])
{
    // Padding the input array with repetitions of the first (padded_input_col_size - input_col_size) columns at the end
    // Now padding the array with repetitions of the first (padded_input_row_size - input_row) rows at the bottom

    //std::cout << "output size:[" << output_row_size << "][" << output_col_size << std::endl;
    register int row, col, temp_col;
    register int input_row_half_addr, output_row_half_addr;
    register int input_row_half_addr2, output_row_half_addr2;
    register int limit_row = (input_row_size - 1) / 2;
    register int limit_col = (input_col_size - 1) / 2;
    input_row_half_addr = 0; //row_offset * input_col_size;
    output_row_half_addr = 0;
    input_row_half_addr2 = input_col_size;
    output_row_half_addr2 = 0;
    for (row = 0; row <= limit_row; ++row)
    {
        col = 0; temp_col = 0; //col_offset;
        for(; col < limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr + temp_col, input, output_row_half_addr + col, outputs[0], outputs[1]);
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr2 + temp_col, input, output_row_half_addr2 + col, outputs[2], outputs[3]);
        }
        /*for (; col <= limit_col - 4; col += 4, temp_col += 8)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[0][output_row_half_addr + col + 1] = (short int)input[input_row_half_addr + temp_col + 2];
            outputs[0][output_row_half_addr + col + 2] = (short int)input[input_row_half_addr + temp_col + 4];
            outputs[0][output_row_half_addr + col + 3] = (short int)input[input_row_half_addr + temp_col + 6];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[1][output_row_half_addr + col + 1] = (short int)input[input_row_half_addr + temp_col + 3];
            outputs[1][output_row_half_addr + col + 2] = (short int)input[input_row_half_addr + temp_col + 5];
            outputs[1][output_row_half_addr + col + 3] = (short int)input[input_row_half_addr + temp_col + 7];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[2][output_row_half_addr2 + col + 1] = (short int)input[input_row_half_addr2 + temp_col + 2];
            outputs[2][output_row_half_addr2 + col + 2] = (short int)input[input_row_half_addr2 + temp_col + 4];
            outputs[2][output_row_half_addr2 + col + 3] = (short int)input[input_row_half_addr2 + temp_col + 6];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            outputs[3][output_row_half_addr2 + col + 1] = (short int)input[input_row_half_addr2 + temp_col + 3];
            outputs[3][output_row_half_addr2 + col + 2] = (short int)input[input_row_half_addr2 + temp_col + 5];
            outputs[3][output_row_half_addr2 + col + 3] = (short int)input[input_row_half_addr2 + temp_col + 7];
            
        }
        */
        for (; col <= limit_col; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        temp_col -= input_col_size;
        for (; col < output_col_size; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
        input_row_half_addr2 += 2 * input_col_size;
        output_row_half_addr2 += output_col_size;
    }
    input_row_half_addr2 -= input_row_size * input_col_size;
    input_row_half_addr -= input_row_size * input_col_size;
    for (; row < output_row_size; ++row)
    {
        col = 0, temp_col = 0; //col_offset;
        for(; col < limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr + temp_col, input, output_row_half_addr + col, outputs[0], outputs[1]);
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr2 + temp_col, input, output_row_half_addr2 + col, outputs[2], outputs[3]);
        }
        for (; col <= limit_col; ++col, temp_col+=2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        temp_col -= input_col_size;
        for (; col < output_col_size; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
        input_row_half_addr2 += 2 * input_col_size;
        output_row_half_addr2 += output_col_size;
    }
}

void createQuarterArraysWithOddOddInput(register int input_row_size, register int input_col_size, register int *input,
                        register int output_row_size, register int output_col_size, register short int *outputs[4])
{
    // Padding the input array with repetitions of the first (padded_input_col_size - input_col_size) columns at the end
    // Now padding the array with repetitions of the first (padded_input_row_size - input_row) rows at the bottom

    //std::cout << "output size:[" << output_row_size << "][" << output_col_size << std::endl;
    register int row, col, temp_col;
    register int input_row_half_addr, output_row_half_addr;
    register int input_row_half_addr2, output_row_half_addr2;
    register int limit_row = (input_row_size - 1) / 2;
    register int limit_col = (input_col_size - 1) / 2;
    input_row_half_addr = 0; //row_offset * input_col_size;
    output_row_half_addr = 0;
    input_row_half_addr2 = input_col_size;
    output_row_half_addr2 = 0;
    for (row = 0; row < limit_col; ++row)
    {
        col = 0; temp_col = 0; //col_offset;
        for(; col < limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr + temp_col, input, output_row_half_addr + col, outputs[0], outputs[1]);
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr2 + temp_col, input, output_row_half_addr2 + col, outputs[2], outputs[3]);
        }
        /*for (; col <= limit_col - 4; col += 4, temp_col += 8)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[0][output_row_half_addr + col + 1] = (short int)input[input_row_half_addr + temp_col + 2];
            outputs[0][output_row_half_addr + col + 2] = (short int)input[input_row_half_addr + temp_col + 4];
            outputs[0][output_row_half_addr + col + 3] = (short int)input[input_row_half_addr + temp_col + 6];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[1][output_row_half_addr + col + 1] = (short int)input[input_row_half_addr + temp_col + 3];
            outputs[1][output_row_half_addr + col + 2] = (short int)input[input_row_half_addr + temp_col + 5];
            outputs[1][output_row_half_addr + col + 3] = (short int)input[input_row_half_addr + temp_col + 7];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[2][output_row_half_addr2 + col + 1] = (short int)input[input_row_half_addr2 + temp_col + 2];
            outputs[2][output_row_half_addr2 + col + 2] = (short int)input[input_row_half_addr2 + temp_col + 4];
            outputs[2][output_row_half_addr2 + col + 3] = (short int)input[input_row_half_addr2 + temp_col + 6];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            outputs[3][output_row_half_addr2 + col + 1] = (short int)input[input_row_half_addr2 + temp_col + 3];
            outputs[3][output_row_half_addr2 + col + 2] = (short int)input[input_row_half_addr2 + temp_col + 5];
            outputs[3][output_row_half_addr2 + col + 3] = (short int)input[input_row_half_addr2 + temp_col + 7];
            
        }
        */
        for (; col < limit_col; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        for (; col <= limit_col; ++col, temp_col+=2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1 - input_col_size];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1 - input_col_size];
            
        }
        temp_col -= input_col_size;
        for (; col < output_col_size; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
        input_row_half_addr2 += 2 * input_col_size;
        output_row_half_addr2 += output_col_size;
    }
    input_row_half_addr2 -= input_row_size * input_col_size;
    //std::cout << row ;
    for (; row <= limit_row; ++row)
    {
        //std::cout << "hello:";
        col = 0; temp_col = 0; //col_offset;
        for(; col < limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr + temp_col, input, output_row_half_addr + col, outputs[0], outputs[1]);
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr2 + temp_col, input, output_row_half_addr2 + col, outputs[2], outputs[3]);
        }
        for (; temp_col < (input_col_size-1); ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        for (; col <= limit_col; ++col, temp_col+=2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1 - input_col_size];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1 - input_col_size];
            
        }
        temp_col -= input_col_size;
        for (; col < output_col_size; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
        input_row_half_addr2 += 2 * input_col_size;
        output_row_half_addr2 += output_col_size;
    }

    input_row_half_addr -= input_row_size * input_col_size;
    for (; row < output_row_size; ++row)
    {
        col = 0, temp_col = 0; //col_offset;
        for(; col < limit_col - 8; col+= 8, temp_col += 16) {
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr + temp_col, input, output_row_half_addr + col, outputs[0], outputs[1]);
            copy8IntTo8ShortIntWith1DilationTo2Arrays(input_row_half_addr2 + temp_col, input, output_row_half_addr2 + col, outputs[2], outputs[3]);
        }
        for (; temp_col < (input_col_size-1); ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        for (; col <= limit_col; ++col, temp_col+=2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1 - input_col_size];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1 - input_col_size];
            
        }
        temp_col -= input_col_size;
        for (; col < output_col_size; ++col, temp_col += 2)
        {
            outputs[0][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col];
            outputs[1][output_row_half_addr + col] = (short int)input[input_row_half_addr + temp_col + 1];
            outputs[2][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col];
            outputs[3][output_row_half_addr2 + col] = (short int)input[input_row_half_addr2 + temp_col + 1];
            
        }
        input_row_half_addr += 2 * input_col_size;
        output_row_half_addr += output_col_size;
        input_row_half_addr2 += 2 * input_col_size;
        output_row_half_addr2 += output_col_size;
    }
}

force_inline int getNumCores()
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

static force_inline int multiplyAndReduce16bit16Integers(const short int *input, const short int *kernel,
                                                         int input_row_half_addr, int input_col,
                                                         int kernel_row_half_addr, int kernel_col)
{
    __m256i vector1 = _mm256_loadu_si256((__m256i *)&input[input_row_half_addr + input_col]);
    __m256i vector2 = _mm256_loadu_si256((__m256i *)&kernel[kernel_row_half_addr + kernel_col]);
    __m256i result = _mm256_mullo_epi16(vector1, vector2);

    // Convert the 16-bit 16 integers to 32-bit 16 integers
    __m256i low128_256 =
        _mm256_cvtepu16_epi32((__m128i)_mm256_castsi256_si128(result)); // Extract lower 128 bits (8 integers)
    __m256i high128_256 =
        _mm256_cvtepu16_epi32((__m128i)_mm256_extracti128_si256(result, 1)); // Extract higher 128 bits (8 integers)
    __m256i accumulated = _mm256_add_epi32(low128_256, high128_256);         // Accumulate the 32-bit 8 integers

    __m128i low128 = _mm256_castsi256_si128(accumulated);       // Extract lower 128 bits (4 integers)
    __m128i high128 = _mm256_extracti128_si256(accumulated, 1); // Extract higher 128 bits (4 integers)
    __m128i sum128 = _mm_add_epi32(low128, high128);            // Add the two 128-bit results
    sum128 = _mm_hadd_epi32(sum128, sum128);                    // Perform horizontal addition

    // Access the result
    int32_t *resultArray = (int32_t *)&sum128;

    // Extract the first two 32-bit integers and add them
    int32_t sum = resultArray[0] + resultArray[1];

    return sum;
}

static force_inline void multiplyAndReduce16bit16IntegersTwoRows(const short int *input, const short int *kernel,
                                                         int input_row_half_addr1, int input_row_half_addr2, int input_col,
                                                         int kernel_row_half_addr, int kernel_col,
                                                         ull &sum1, ull &sum2)
{
    __m256i vector1 = _mm256_loadu_si256((__m256i *)&input[input_row_half_addr1 + input_col]);
    __m256i vector2 = _mm256_loadu_si256((__m256i *)&kernel[kernel_row_half_addr + kernel_col]);
    __m256i vector3 = _mm256_loadu_si256((__m256i *)&input[input_row_half_addr2 + input_col]);
    __m256i result1 = _mm256_mullo_epi16(vector1, vector2);
    __m256i result2 = _mm256_mullo_epi16(vector3, vector2);

    // Convert the 16-bit 16 integers to 32-bit 16 integers
    __m256i low128_256_1 =
        _mm256_cvtepu16_epi32((__m128i)_mm256_castsi256_si128(result1)); // Extract lower 128 bits (8 integers)
    __m256i low128_256_2 =
        _mm256_cvtepu16_epi32((__m128i)_mm256_castsi256_si128(result2)); // Extract lower 128 bits (8 integers)
    __m256i high128_256_1 =
        _mm256_cvtepu16_epi32((__m128i)_mm256_extracti128_si256(result1, 1)); // Extract higher 128 bits (8 integers)
    __m256i high128_256_2 =
        _mm256_cvtepu16_epi32((__m128i)_mm256_extracti128_si256(result2, 1)); // Extract higher 128 bits (8 integers)
    __m256i accumulated_1 = _mm256_add_epi32(low128_256_1, high128_256_1);         // Accumulate the 32-bit 8 integers
    __m256i accumulated_2 = _mm256_add_epi32(low128_256_2, high128_256_2);         // Accumulate the 32-bit 8 integers

    __m128i low128_1 = _mm256_castsi256_si128(accumulated_1);       // Extract lower 128 bits (4 integers)
    __m128i low128_2 = _mm256_castsi256_si128(accumulated_2);       // Extract lower 128 bits (4 integers)
    __m128i high128_1 = _mm256_extracti128_si256(accumulated_1, 1); // Extract higher 128 bits (4 integers)
    __m128i high128_2 = _mm256_extracti128_si256(accumulated_2, 1); // Extract higher 128 bits (4 integers)
    __m128i sum128_1 = _mm_add_epi32(low128_1, high128_1);            // Add the two 128-bit results
    __m128i sum128_2 = _mm_add_epi32(low128_2, high128_2);            // Add the two 128-bit results
    sum128_1 = _mm_hadd_epi32(sum128_1, sum128_1);                    // Perform horizontal addition
    sum128_2 = _mm_hadd_epi32(sum128_2, sum128_2);                    // Perform horizontal addition

    // Access the result
    int32_t *resultArray = (int32_t *)&sum128_1;
    int32_t *resultArray2 = (int32_t *)&sum128_2;

    // Extract the first two 32-bit integers and add them
    sum1 += resultArray[0] + resultArray[1];
    sum2 += resultArray2[0] + resultArray2[1];

}

static force_inline int multiplyAndReduce16bit8Integers(const short int *input, const short int *kernel,
                                                        int input_row_half_addr, int input_col,
                                                        int kernel_row_half_addr, int kernel_col)
{
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

    return sum;
}

static force_inline void multiplyAndReduce16bit8IntegersTwoRows(const short int *input, const short int *kernel,
                                                        int input_row_half_addr, int input_row_half_addr2, int input_col,
                                                        int kernel_row_half_addr, int kernel_col, ull& sum1, ull& sum2)
{
    __m128i vector1 = _mm_loadu_si128((__m128i *)&input[input_row_half_addr + input_col]);
    __m128i vector2 = _mm_loadu_si128((__m128i *)&kernel[kernel_row_half_addr + kernel_col]);
    __m128i vector3 = _mm_loadu_si128((__m128i *)&input[input_row_half_addr2 + input_col]);
    __m128i result_1 = _mm_mullo_epi16(vector1, vector2);
    __m128i result_2 = _mm_mullo_epi16(vector3, vector2);

    // Convert the 16-bit integers to 32-bit integers
    __m128i lowPart_1 = _mm_cvtepu16_epi32(_mm_unpacklo_epi64(result_1, result_1));
    __m128i lowPart_2 = _mm_cvtepu16_epi32(_mm_unpacklo_epi64(result_2, result_2));
    __m128i highPart_1 = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(result_1, result_1));
    __m128i highPart_2 = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(result_2, result_2));

    // Add the extended values
    __m128i sum128_1 = _mm_add_epi32(lowPart_1, highPart_1);
    __m128i sum128_2 = _mm_add_epi32(lowPart_2, highPart_2);

    sum128_1 = _mm_hadd_epi32(sum128_1, sum128_1); // Perform horizontal addition
    sum128_2 = _mm_hadd_epi32(sum128_2, sum128_2); // Perform horizontal addition

    // Access the result
    int32_t *resultArray1 = (int32_t *)&sum128_1;
    int32_t *resultArray2 = (int32_t *)&sum128_2;

    // Extract the first two 32-bit integers and add them
    sum1 += resultArray1[0] + resultArray1[1];
    sum2 += resultArray2[0] + resultArray2[1];
}

static force_inline int multiplyAndReduce16bit4Integers(const short int *input, const short int *kernel,
                                                        int input_row_half_addr, int input_col,
                                                        int kernel_row_half_addr, int kernel_col)
{
    __m128i vector1 = _mm_loadu_si128((__m128i *)&input[input_row_half_addr + input_col]);
    __m128i vector2 = _mm_loadu_si128((__m128i *)&kernel[kernel_row_half_addr + kernel_col]);
    __m128i result = _mm_mullo_epi16(vector1, vector2);

    // Convert the 16-bit integers to 32-bit integers
    result = _mm_cvtepu16_epi32(_mm_unpacklo_epi64(result, result));
    //__m128i highPart = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(result, result));

    // Add the extended values
    //__m128i sum128 = _mm_add_epi32(lowPart, highPart);

    __m128i sum128 = _mm_hadd_epi32(result, result); // Perform horizontal addition

    // Access the result
    int32_t *resultArray = (int32_t *)&sum128;

    // Extract the first two 32-bit integers and add them
    int32_t sum = resultArray[0] + resultArray[1];

    // output[output_row_half_addr + output_col] += sum;
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

static force_inline void multiplyAndReduce16bit4IntegersTwoRows(const short int *input, const short int *kernel,
                                                        int input_row_half_addr, int input_row_half_addr2, int input_col,
                                                        int kernel_row_half_addr, int kernel_col,
                                                        ull& sum1, ull& sum2)
{
    __m128i vector1 = _mm_loadu_si128((__m128i *)&input[input_row_half_addr + input_col]);
    __m128i vector2 = _mm_loadu_si128((__m128i *)&kernel[kernel_row_half_addr + kernel_col]);
    __m128i vector3 = _mm_loadu_si128((__m128i *)&input[input_row_half_addr2 + input_col]);
    __m128i result_1 = _mm_mullo_epi16(vector1, vector2);
    __m128i result_2 = _mm_mullo_epi16(vector3, vector2);

    // Convert the 16-bit integers to 32-bit integers
    result_1 = _mm_cvtepu16_epi32(_mm_unpacklo_epi64(result_1, result_1));
    result_2 = _mm_cvtepu16_epi32(_mm_unpacklo_epi64(result_2, result_2));

    result_1 = _mm_hadd_epi32(result_1, result_1); // Perform horizontal addition
    result_2 = _mm_hadd_epi32(result_2, result_2); // Perform horizontal addition

    // Access the result
    int32_t *resultArray1 = (int32_t *)&result_1;
    int32_t *resultArray2 = (int32_t *)&result_2;

    // Extract the first two 32-bit integers and add them
    sum1 += resultArray1[0] + resultArray1[1];
    sum2 += resultArray2[0] + resultArray2[1];
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
