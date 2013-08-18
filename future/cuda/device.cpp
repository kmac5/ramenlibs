//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#include<base/cuda/device.hpp>

#include<base/cuda/cuda.hpp>

namespace base
{
namespace cuda
{

void get_devices( std::vector<device_t>& devices)
{
    devices.clear();
    int num_devs = cu_device_get_count();

    char tmp_name[256];

    for( int i = 0; i < num_devs; ++i)
    {
        device_t dev;
        dev.number = i;
        cu_device_get_name( tmp_name, 256, i);
        dev.name = tmp_name;

        cu_device_compute_capability( &dev.compute_major, &dev.compute_minor, i);
        dev.total_global_mem = cu_device_total_mem( i);
        dev.multi_processor_count = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, i);
        dev.clock_rate = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_CLOCK_RATE, i);
        dev.memory_clock = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, i);
		dev.mem_bus_width = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, i);
        dev.l2_cache_size = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, i);

		dev.max_tex1d = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, i);
		dev.max_tex2d[0] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, i);
		dev.max_tex2d[1] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, i);
		dev.max_tex3d[0] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, i);
		dev.max_tex3d[1] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, i);
		dev.max_tex3d[2] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, i);

        dev.max_tex2d_layered[0] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, i);
        dev.max_tex2d_layered[1] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, i);
        dev.max_tex2d_layered[2] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, i);

		dev.total_constant_memory = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, i);
		dev.shared_mem_per_block = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, i);
		dev.regs_per_block = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, i);
		dev.warp_size = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_WARP_SIZE, i);
		dev.max_threads_per_multi_processor = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, i);
		dev.max_threads_per_block = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, i);

		dev.block_dim[0] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, i);
		dev.block_dim[1] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, i);
		dev.block_dim[2] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, i);

		dev.grid_dim[0] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, i);
		dev.grid_dim[1] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, i);
		dev.grid_dim[2] = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, i);

        dev.texture_align = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, i);
		dev.mem_pitch = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_MAX_PITCH, i);
        dev.gpu_overlap = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, i);
        dev.async_engine_count = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, i);

        dev.kernel_exec_timeout_enabled = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, i);
        dev.integrated = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_INTEGRATED, i);
        dev.can_map_host_memory = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, i);

        dev.concurrent_kernels = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, i);
        dev.surface_alignment = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, i);
        dev.ecc_enabled = cu_device_get_attribute(  CU_DEVICE_ATTRIBUTE_ECC_ENABLED, i);

        dev.tcc_driver = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_TCC_DRIVER, i);

        dev.unified_addressing = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, i);
        dev.pci_bus_id = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, i);
        dev.pci_device_id = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, i);
        dev.compute_mode = cu_device_get_attribute( CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, i);

        devices.push_back( dev);
    }
}

} // namespace
} // namespace

/*
        error_id =  cu_device_compute_capability(&major, &minor, dev);
		if (error_id != _c_u_d_a_s_u_c_c_e_s_s) {
			shr_log( "cu_device_compute_capability returned %d\n-> %s\n", (int)error_id, get_cuda_drv_error_string(error_id) );
			shr_q_a_finish_exit(argc, (const char **)argv, _q_a_f_a_i_l_e_d);
		}

        shr_log("  _c_u_d_a _capability _major/_minor version number:    %d.%d\n", major, minor);

        size_t total_global_mem;
        error_id = cu_device_total_mem(&total_global_mem, dev);
		if (error_id != _c_u_d_a_s_u_c_c_e_s_s) {
			shr_log( "cu_device_total_mem returned %d\n-> %s\n", (int)error_id, get_cuda_drv_error_string(error_id) );
			shr_q_a_finish_exit(argc, (const char **)argv, _q_a_f_a_i_l_e_d);
		}

        char msg[256];
        sprintf(msg, "  _total amount of global memory:                 %.0f _m_bytes (%llu bytes)\n",
                      (float)total_global_mem/1048576.0f, (unsigned long long) total_global_mem);
        shr_log(msg);
*/


