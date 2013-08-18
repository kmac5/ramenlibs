//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#ifndef BASE_CUDA_DEVICE_HPP
#define BASE_CUDA_DEVICE_HPP

#include<base/config.hpp>

#include<string>
#include<vector>

namespace base
{
namespace cuda
{

struct BASE_API device_t
{
    unsigned int number;
    std::string name;
    int compute_major, compute_minor;
    size_t total_global_mem;
    int multi_processor_count;
    int clock_rate;
    int memory_clock;
    int mem_bus_width;
    int l2_cache_size;
    int max_tex1d, max_tex2d[2], max_tex3d[3];
    int  max_tex2d_layered[3];
    int total_constant_memory;
    int shared_mem_per_block;
    int regs_per_block;
    int warp_size;
    int max_threads_per_multi_processor;
    int max_threads_per_block;
    int block_dim[3];
    int grid_dim[3];
    int texture_align;
    int mem_pitch;
    int gpu_overlap;
    int async_engine_count;
    int kernel_exec_timeout_enabled;
    int integrated;
    int can_map_host_memory;
    int concurrent_kernels;
    int surface_alignment;
    int ecc_enabled;
    int tcc_driver ;
    int unified_addressing;
    int pci_bus_id, pci_device_id;
    int compute_mode;
};

BASE_API void get_devices( std::vector<device_t>& devices);

} // namespace
} // namespace

#endif
