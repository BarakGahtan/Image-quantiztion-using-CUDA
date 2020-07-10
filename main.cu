///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex1.h"

#define SQR(a) ((a) * (a))

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

int randomize_images(uchar *images)
{
    FILE *urandom = fopen("/dev/urandom", "r");
    if (!urandom) {
	perror("Unable to open /dev/urandom");
	return 1;
    }

    int ret = fread(images, 1, N_IMAGES * IMG_WIDTH * IMG_HEIGHT, urandom);
    if (ret < N_IMAGES * IMG_HEIGHT * IMG_WIDTH) {
	perror("Unable to read random data");
	fclose(urandom);
	return 1;
    }
    fclose(urandom);
    return 0;
} 

int main() {
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    int devices;
    CUDA_CHECK( cudaGetDeviceCount(&devices) );
    printf("Number of devices: %d\n", devices);

    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    double t_start, t_finish;

    /* instead of loading real images, we'll load the arrays with random data */
    printf("\n=== Randomizing images ===\n");
    t_start = get_time_msec();
    if (randomize_images(images_in))
	return 1;
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        cpu_process(img_in, img_out, IMG_WIDTH, IMG_HEIGHT);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n");

    struct task_serial_context *ts_context = task_serial_init();

    t_start = get_time_msec();
    task_serial_process(ts_context, images_in, images_out_gpu_serial);
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial);
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr);

    task_serial_free(ts_context);

    // GPU bulk
    printf("\n=== GPU Bulk ===\n");
    struct gpu_bulk_context *gb_context = gpu_bulk_init();
    t_start = get_time_msec();
    gpu_bulk_process(gb_context, images_in, images_out_gpu_bulk);
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk);
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr);

    gpu_bulk_free(gb_context);

    return 0;
}
