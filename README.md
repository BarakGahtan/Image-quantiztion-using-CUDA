# Image-quantiztion-using-CUDA
In this assignment we will implement image quantization for grayscale images. This method reduces the number of colors used in an image.  The algorithm we will implement works by first creating a histogram of the gray levels in the image, then using the histogram to create a map which maps each value of these levels to a new value, and finally, using this map to create the new image.  We will implement quantization on MxN grayscale images, each represented as a unsigned char array of length M*N, with values in the range [0, 255], where 0 means black, and 255 means white.

IN THE HOMEWORK1 PDF the full details. 
