# cuda-cpp-bilateral-filter
Profiling performance of bilateral filtering on 2D image between OpenCV and my implementation.
My implementation use CUDA's texture memory to access the input image pixel instead of using global memory.

## Dependencies (Which I tested on)
Make sure that OpenCV is compiled with cuda functionalities.
Ubuntu == 18.04
CUDA == 10.2
GCC/G++ == 8.4.0
OpenCV == 4.2.0
cxxopts == 2.2.1 

## Build and Execute 
```
nvcc src/kernel.cu src/main.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc
./a.out -i data/depth_map.png -o filtered_depth_map.png
```

## List of trying to figure out.
- Why memory copy using cv::cuda::GpuMat is so slow?
