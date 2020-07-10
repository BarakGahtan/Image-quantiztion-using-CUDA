#include "ex1.h"

/*--------------------------------------------------------------------------------
                                        AUX
--------------------------------------------------------------------------------*/
__device__ void calculate_histogram(uchar *given_image, int *result_array, int len) {
    int thread_id = threadIdx.x;
    /*amount of pixels that each thread does over the image. */
    int work_per_thread = (IMG_WIDTH * IMG_HEIGHT) / blockDim.x;
    
    /* initialize the histogram */
    if (thread_id < 256) {
        result_array[thread_id] = 0;
    }
    __syncthreads();

    /* calculate the histogram */
    int index;
    for (int i=0 ; i < work_per_thread ; i++) {
        index = blockDim.x * i + thread_id;
        atomicAdd(&result_array[given_image[index]], 1);
    }
    __syncthreads();
}
__device__ void compute_map(int *cdf, uchar* map) {

    int thread_id = threadIdx.x;

    if (thread_id < 256) {
        map[thread_id] = (256/N_COLORS) * floorf((N_COLORS *cdf[thread_id])/(IMG_HEIGHT * IMG_WIDTH));
    }
    __syncthreads();
}

__device__ void remap_image(uchar *in_image, uchar *out_image, uchar *map) {

    int thread_id = threadIdx.x;
    int work_per_thread = (IMG_WIDTH * IMG_HEIGHT) / blockDim.x;
    
    /* remap the image */
    int index;
    for (int i=0 ; i<work_per_thread ; i++) {
        index = blockDim.x * i + thread_id;
        out_image[index] = map[in_image[index]];
    }
    __syncthreads();
}
__device__ void prefix_sum(int arr[], int arr_size) {
    int thread_id = threadIdx.x;
    int addition;
    for (int stride = 1; stride < blockDim.x; stride <<= 1){
        //for each thread computes the value, locally and wait until the rest finished.
        if (thread_id >= stride && thread_id - stride < arr_size)
            addition = arr[thread_id - stride];
        
        //use barrier to wait
        __syncthreads();
        
        // write back to global memory once computation is done
        if (thread_id >= stride && thread_id < arr_size)
            arr[thread_id] += addition;
        
        __syncthreads();
    }
}

__device__ void process_image_wrapper(uchar *in, uchar *out) {
    __shared__ int cdf[256];
    __shared__ uchar m[256];
    calculate_histogram(in, cdf, 256);
    prefix_sum(cdf, 256);
    compute_map(cdf, m);
    remap_image(in, out, m);
}

__global__ void process_image_kernel(uchar *in, uchar *out) {
    process_image_wrapper(in, out);
}

// process all images
__global__ void process_all_images_kernel(uchar *all_in, uchar *all_out) {
    int block_id = blockIdx.x;
    int offset = block_id * IMG_WIDTH * IMG_HEIGHT;
    process_image_wrapper(all_in + offset, all_out + offset);
}


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar* image_in;
    uchar* image_out;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    //allocate memeory on the GPU global memory
    auto context = new task_serial_context;
    cudaMalloc(&context->image_in, IMG_WIDTH * IMG_HEIGHT);
    cudaMalloc(&context->image_out, IMG_WIDTH * IMG_HEIGHT);
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out){
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial

    for (int i=0 ; i < N_IMAGES ; i++) {
        uchar *image_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *image_out = &images_out[i * IMG_WIDTH * IMG_HEIGHT];
        /* copy the relevant image from images_in to the GPU memory allocated */
        cudaMemcpy(context->image_in, image_in, IMG_WIDTH * IMG_HEIGHT,cudaMemcpyHostToDevice);

        int blocks = 1;
        int threads_in_block = 1024;
        /* invoke the GPU kernel */
        process_image_kernel<<<blocks, threads_in_block>>>(context->image_in, context->image_out);
        
        cudaMemcpy(image_out, context->image_out , IMG_WIDTH * IMG_HEIGHT,cudaMemcpyDeviceToHost);
    }

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    cudaFree(context->image_in);
    cudaFree(context->image_out);
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    uchar* all_in;
    uchar* all_out;
};

/* Allocate GPU memory for all the input and output images.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;
    cudaMalloc(&context->all_in, IMG_WIDTH * IMG_HEIGHT * N_IMAGES);
    cudaMalloc(&context->all_out, IMG_WIDTH * IMG_HEIGHT * N_IMAGES);
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out

    /* copy the relevant images from images_in to the GPU memory allocated */
    cudaMemcpy(context->all_in, images_in, IMG_WIDTH * IMG_HEIGHT * N_IMAGES ,cudaMemcpyHostToDevice);
    int blocks = N_IMAGES;
    int threads_in_block = 1024;
    /* invoke the GPU kernel */
    process_all_images_kernel<<<blocks, threads_in_block>>>(context->all_in, context->all_out);
    /* copy output from GPU memory to relevant location in images_out_gpu_serial */
    cudaMemcpy(images_out, context->all_out , IMG_WIDTH * IMG_HEIGHT * N_IMAGES,cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    cudaFree(context->all_in);
    cudaFree(context->all_out);
    free(context);
}
