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

#ifndef RAMEN_CUDA_DEVICE_SEARCH_HPP
#define RAMEN_CUDA_DEVICE_SEARCH_HPP

#include<ramen/cuda/cudart.hpp>

#include<boost/range/algorithm/find_if.hpp>

namespace ramen
{
namespace cuda
{

struct select_max_cores_device
{
    select_max_cores_device()
    {
        device = -1;
        num_cores = -1;
    }

    void operator()( int device_num, const cudaDeviceProp& props)
    {
        if( num_cores < props.multiProcessorCount)
        {
            device = device_num;
            num_cores = props.multiProcessorCount;
        }
    }

    int device;
    int num_cores;
};

struct select_max_memory_device
{
    select_max_memory_device()
    {
        device = -1;
        mem_size = 0;
    }

    void operator()( int device_num, const cudaDeviceProp& props)
    {
        if( mem_size < props.totalGlobalMem)
        {
            device = device_num;
            mem_size = props.totalGlobalMem;
        }
    }

    int device;
    std::size_t mem_size;
};


template<class SearchFun>
void device_search( SearchFun& f)
{
    cudaDeviceProp props;

    for( int i = 0, e = cuda_get_device_count(); i < e; ++i)
    {
        cuda_get_device_properties( &props, i);
        f( i, props);
    }
}

} // cuda
} // ramen

#endif
