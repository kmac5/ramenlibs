/*
 Copyright (c) 2013 Esteban Tovagliari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include<iostream>

#include<boost/cstdlib.hpp>
#include<boost/scoped_array.hpp>

#include<ramen/cuda/cudart.hpp>
#include<ramen/cuda/auto_ptr.hpp>

using namespace ramen::cuda;

__global__ void add_vectors( int n, float *a, float *b, float *c)
{
    int tid = blockIdx.x;
    if( tid < n)
        c[tid] = a[tid] + b[tid];
}

int main(int argc, char **argv)
{
    try
    {
        int devices = cuda_get_device_count();

        if( !devices)
        {
            std::cerr << "No CUDA devices found. Exiting" << std::endl;
            return boost::exit_failure;
        }

        boost::scoped_array<float> src( new float[10]);
        for( int i = 0; i < 10; ++i)
            src[i] = i;

        auto_ptr_t<float,device_ptr_policy> gpu_src1( cuda_malloc<float>( 10));
        cuda_memcpy( gpu_src1.get(), src.get(), 10, cudaMemcpyHostToDevice);

        auto_ptr_t<float,device_ptr_policy> gpu_src2( cuda_malloc<float>( 10));
        cuda_memcpy( gpu_src2.get(), src.get(), 10, cudaMemcpyHostToDevice);

        auto_ptr_t<float,device_ptr_policy> gpu_dst( cuda_malloc<float>( 10));

        add_vectors<<<10,1>>>( 10, gpu_src1.get(), gpu_src2.get(), gpu_dst.get());

        boost::scoped_array<float> dst( new float[10]);
        cuda_memcpy( dst.get(), gpu_dst.get(), 10, cudaMemcpyDeviceToHost);

        for( int i = 0; i < 10; ++i)
        {
            if( dst[i] != ( i + i))
                return boost::exit_failure;
        }

        return boost::exit_success;
    }
    catch( ramen::core::exception& e)
    {
        std::cerr << "exception caught, " << e.what() << std::endl;
        return boost::exit_failure;
    }
}
