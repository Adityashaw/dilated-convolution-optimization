
#include "utility.h"
#include <iostream>
#include <pthread.h>

#define ull unsigned long long int

// structure to pass data to thread
struct ThreadData
{
    int thread_id;
    int num_threads;
    int input_row_size;
    int input_col_size;
    short int *input;
    short int *kernel;
    int kernel_row_size;
    int kernel_col_size;
    int output_row_size;
    int output_col_size;
    ull *output;
};

// wrapper structure to pass data to thread
struct WrapperThreadData
{
    int thread_id;
    int num_threads;
    int input_row_size;
    int input_col_size;
    int *input;
    short int *kernel;
    int kernel_row_size;
    int kernel_col_size;
    int output_row_size;
    int output_col_size;
    ull *output;
};

// actual thread function which performs convolution
void *threadInsideQuarterOutput(void *arg)
{
    ThreadData *data = static_cast<ThreadData *>(arg);

    // Local variables to read from the thread data
    int thread_id = data->thread_id;
    int num_threads = data->num_threads;
    int input_row_size = data->input_row_size;
    int input_col_size = data->input_col_size;
    short int *input = data->input;
    short int *kernel = data->kernel;
    int kernel_row_size = data->kernel_row_size;
    int kernel_col_size = data->kernel_col_size;
    int output_row_size = data->output_row_size;
    int output_col_size = data->output_col_size;
    ull *output = data->output;

    // Calculate the range of rows for this thread
    int rows_per_thread = output_row_size / num_threads;
    int output_row_start = thread_id * rows_per_thread;
    int output_row_end = (thread_id == num_threads - 1) ? output_row_size : output_row_start + rows_per_thread;

    convolute(input_row_size, input_col_size, input, kernel_row_size, kernel_col_size, kernel, output_row_start,
              output_row_end, output_col_size, output);

    pthread_exit(NULL);
}

// thread to create quarter input and new threads for convolution
void *threadForQuarterOutput(void *arg)
{
    WrapperThreadData *data = static_cast<WrapperThreadData *>(arg);

    // Local variables to read from the thread data
    int thread_id = data->thread_id;
    int input_row_size = data->input_row_size;
    int input_col_size = data->input_col_size;
    int *input = data->input;
    short int *kernel = data->kernel;
    int kernel_row_size = data->kernel_row_size;
    int kernel_col_size = data->kernel_col_size;
    int output_row_size = data->output_row_size;
    int output_col_size = data->output_col_size;
    ull *output = data->output;

    int num_cores = getNumCores();

    if (num_cores == 0)
    {
        // Unable to determine the number of cores
        std::cerr << "Unable to determine the number of cores. Defaulting to a reasonable value." << std::endl;
        num_cores = 4; // Default to 4 cores
    }
    num_cores /= 4;

    int num_threads = std::max(1, num_cores);
    pthread_t threads[num_threads];
    // create one of the four sub-blocks of input with padding
    int padded_input_row_size = input_row_size + 1 * kernel_row_size;
    int padded_input_col_size = input_col_size + 1 * kernel_col_size;
    int quarter_input_row_sizes[4] = {(padded_input_row_size + 1) / 2, (padded_input_row_size + 1) / 2,
                                      (padded_input_row_size) / 2, (padded_input_row_size) / 2};
    int quarter_input_col_sizes[4] = {(padded_input_col_size + 1) / 2, (padded_input_col_size) / 2,
                                      (padded_input_col_size + 1) / 2, (padded_input_col_size) / 2};

    short int *quarter_input =
        new short int[quarter_input_row_sizes[thread_id] *
                      quarter_input_col_sizes[thread_id]]; // new array to store each of the input matrix with padding.
    register int row_offset = thread_id >> 1;
    register int col_offset = thread_id & 1;
    createQuarterArray(input_row_size, input_col_size, input, quarter_input_row_sizes[thread_id],
                       quarter_input_col_sizes[thread_id], quarter_input, row_offset,
                       col_offset); // fill one of the four subblocks of the input matrix with padding

    for (int thread_i = 0; thread_i < num_threads; ++thread_i)
    {
        ThreadData *thread_data = new ThreadData;
        thread_data->thread_id = thread_i;
        thread_data->num_threads = num_threads;
        thread_data->input_row_size = quarter_input_row_sizes[thread_id];
        thread_data->input_col_size = quarter_input_col_sizes[thread_id];
        thread_data->input = quarter_input;
        thread_data->kernel = kernel;
        thread_data->kernel_row_size = kernel_row_size;
        thread_data->kernel_col_size = kernel_col_size;
        thread_data->output_row_size = output_row_size;
        thread_data->output_col_size = output_col_size;
        thread_data->output = output;

        // Create threads
        int rc = pthread_create(&threads[thread_i], NULL, threadInsideQuarterOutput, static_cast<void *>(thread_data));
        if (rc)
        {
            std::cerr << "Error creating thread: " << rc << std::endl;
            exit(-1);
        }
    }

    // wait for all threads to finish
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    delete[] quarter_input;

    pthread_exit(NULL);
}

void multiThread(int input_row_size, int input_col_size, int *input, int kernel_row_size, int kernel_col_size,
                 int *kernel, int output_row_size, int output_col_size, ull *output)
{
    int output_row_half_addr, output_col_half_addr;
    int partial_output_row_half_addr, partial_output_col_half_addr;
    int row_offset, col_offset;

    short int *kernel2 = new short int[kernel_row_size * kernel_col_size];
    copyKernel(kernel_row_size, kernel_col_size, kernel, kernel2);

    ull *quarter_outputs[4];
    int quarter_output_row_sizes[4] = {(output_row_size + 1) / 2, (output_row_size + 1) / 2, (output_row_size) / 2,
                                       (output_row_size) / 2};
    int quarter_output_col_sizes[4] = {(output_col_size + 1) / 2, (output_col_size) / 2, (output_col_size + 1) / 2,
                                       (output_col_size) / 2};

    int num_threads = 4;
    pthread_t threads[num_threads];

    // create four sub-blocks of output
    for (int i = 0; i < 4; ++i)
    {
        quarter_outputs[i] = new ull[quarter_output_row_sizes[i] * quarter_output_col_sizes[i]](); // new array to st
        WrapperThreadData *thread_data = new WrapperThreadData;
        thread_data->thread_id = i;
        thread_data->num_threads = 4;
        thread_data->input_row_size = input_row_size;
        thread_data->input_col_size = input_col_size;
        thread_data->input = input;
        thread_data->kernel = kernel2;
        thread_data->kernel_row_size = kernel_row_size;
        thread_data->kernel_col_size = kernel_col_size;
        thread_data->output_row_size = quarter_output_row_sizes[i];
        thread_data->output_col_size = quarter_output_col_sizes[i];
        thread_data->output = quarter_outputs[i];

        int rc = pthread_create(&threads[i], NULL, threadForQuarterOutput, static_cast<void *>(thread_data));
        if (rc)
        {
            std::cerr << "Error creating thread: " << rc << std::endl;
            std::exit(-1);
        }
    }

    // Join threads
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    // merge back the four output blocks
    for (int i = 0; i < 4; ++i)
    {
        row_offset = i >> 1;
        col_offset = i & 1;
        output_row_half_addr = row_offset * output_col_size;
        partial_output_row_half_addr = 0;
        for (int row = 0; row < quarter_output_row_sizes[i]; ++row)
        {
            output_col_half_addr = output_row_half_addr + col_offset;
            partial_output_col_half_addr = partial_output_row_half_addr;
            for (int col = 0; col < quarter_output_col_sizes[i]; ++col)
            {
                output[output_col_half_addr] = quarter_outputs[i][partial_output_col_half_addr];
                output_col_half_addr += 2;
                partial_output_col_half_addr += 1;
            }
            output_row_half_addr += output_col_size << 1;
            partial_output_row_half_addr += quarter_output_col_sizes[i];
        }
        delete[] quarter_outputs[i];
    }
    // printArray(output_row_size, output_col_size, output);
    return;
}
