/*
 * Example program to demonstrate how to use the kernel tuner to tune
 * parameters in the host code of GPU programs, such as the number of 
 * streams, in combination with parameters in the kernel
 */
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#ifndef num_streams
    #define num_streams 1
#endif
#ifndef grid_size_x
    #define grid_size_x 1
#endif
#ifndef grid_size_y
    #define grid_size_y 1
#endif

extern "C" {

#include "convolution.cu"

float convolution_streams(float *output, float *input, float *filter) {

    float *h_output = output;
    float *h_input = input;
    float *h_filter = filter;
    float *d_output;
    float *d_input;
    hipError_t err;

    err = hipMalloc((void **)&d_output, image_width*image_height*sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "Error in hipMalloc: %s\n", hipGetErrorString(err));
    }
    err = hipMemset(d_output, 0, image_width*image_height*sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "Error in hipMemset: %s\n", hipGetErrorString(err));
    }
    err = hipMalloc((void **)&d_input, input_width*input_height*sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "Error in hipMalloc: %s\n", hipGetErrorString(err));
    }

    hipStream_t stream[num_streams];
    hipEvent_t event_htod[num_streams];
    for (int i=0; i<num_streams; i++) {
        err = hipStreamCreate(&stream[i]);
        if (err != hipSuccess) {
            fprintf(stderr, "Error in hipStreamCreate: %s\n", hipGetErrorString(err));
        }
        err = hipEventCreate(&event_htod[i]);
        if (err != hipSuccess) {
            fprintf(stderr, "Error in hipEventCreate: %s\n", hipGetErrorString(err));
        }
    }

    hipEvent_t start;
    err = hipEventCreate(&start);
    if (err != hipSuccess) {
        fprintf(stderr, "Error in hipEventCreate: %s\n", hipGetErrorString(err));
    }

    hipEvent_t stop;
    err = hipEventCreate(&stop);
    if (err != hipSuccess) {
        fprintf(stderr, "Error in hipEventCreate: %s\n", hipGetErrorString(err));
    }

    //make sure there have been no errors
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Error after memory setup in convolution_streams: %s\n", hipGetErrorString(err));
    }

    dim3 threads(block_size_x, block_size_y, block_size_z);
    dim3 grid(grid_size_x, grid_size_y);

    //lines per stream, input data per stream, and border size
    int lps = (image_height / num_streams);
    int dps = lps * input_width;
    int border = border_height * input_width;

    //start timing
    hipDeviceSynchronize();
    hipEventRecord(start, 0);

    err = hipMemcpyToSymbolAsync(HIP_SYMBOL(d_filter), h_filter, filter_width*filter_height*sizeof(float), 0, hipMemcpyHostToDevice, stream[0]);
    if (err != hipSuccess) {
        fprintf(stderr, "Error in hipMemcpyToSymbolAsync: %s\n", hipGetErrorString(err));
    }

    //streamed copy of input data with strict order among streams, stream[0] also copies border
    for (int k=0; k<num_streams; k++) {
        if (k == 0) {
            err = hipMemcpyAsync(d_input, h_input, (border + dps)*sizeof(float), hipMemcpyHostToDevice, stream[k]);
        }
        else {
            err = hipStreamWaitEvent(stream[k], event_htod[k-1], 0);
            if (err != hipSuccess) {
                fprintf(stderr, "Error in hipStreamWaitEvent htod k-1: %s\n", hipGetErrorString(err));
            }
            err = hipMemcpyAsync(d_input +border+k*dps, h_input +border+k*dps, dps*sizeof(float), hipMemcpyHostToDevice, stream[k]);
        }
        if (err != hipSuccess) {
            fprintf(stderr, "Error in hipMemcpyHostToDevice: %s\n", hipGetErrorString(err));
        }

        err = hipEventRecord(event_htod[k], stream[k]);
        if (err != hipSuccess) {
            fprintf(stderr, "Error in hipEventRecord htod: %s\n", hipGetErrorString(err));
        }
    }

    //start the kernel in each stream
    for (int k=0; k<num_streams; k++) {
        convolution_kernel<<<grid, threads, 0, stream[k]>>>(d_output+k*lps*image_width, d_input +k*dps, filter);
    }

    //streamed copy of the output data back to the host
    for (int k=0; k<num_streams; k++) {
        err = hipMemcpyAsync(h_output + k*lps*image_width, d_output + k*lps*image_width, lps*image_width*sizeof(float), hipMemcpyDeviceToHost, stream[k]);
        if (err != hipSuccess) {
            fprintf(stderr, "Error in hipMemcpyDeviceToHost: %s\n", hipGetErrorString(err));
        }
    }    

    //mark the end of the computation
    hipEventRecord(stop, 0);

    //wait for all to finish and get time
    hipDeviceSynchronize();
    float time = 0.0;
    hipEventElapsedTime(&time, start, stop);

    memcpy(output, h_output, image_width*image_height*sizeof(float));

    //cleanup
    hipFree(d_output);
    hipFree(d_input);
    for (int k=0; k<num_streams; k++) {
        hipStreamDestroy(stream[k]);
        hipEventDestroy(event_htod[k]);
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);

    //make sure there have been no errors
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        //this bit is necessary because the Kernel Tuner currently can't decide whether
        //it's OK to silently skip an error or break execution when calling C functions
        const char *error_string = hipGetErrorString(err);
        if (strncmp("too many resources requested for launch", error_string, 10) == 0) {
            time = -1.0;
        } else {
            fprintf(stderr, "Error at the end of convolution_streams: %s\n", error_string);
            exit(1);
        }
    }

    return time;
}



}
