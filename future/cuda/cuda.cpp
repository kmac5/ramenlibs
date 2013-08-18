//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#include<base/cuda/cuda.hpp>

#include<boost/function.hpp>

#include<base/assert.hpp>

#if defined __linux__ || defined __APPLE__
    #include<dlfcn.h>
#else
    #if defined _WIN32
        #error "OS not supported yet."
    #else
        #error "OS not supported yet."
    #endif
#endif

namespace base
{
namespace cuda
{
namespace
{

#if defined __linux__ || defined __APPLE__
    void *cuda_lib_ = 0;
#else
    #if defined _WIN32
        #error "OS not supported yet."
    #else
        #error "OS not supported yet."
    #endif
#endif

boost::function<CUresult ( unsigned int)> cuInit_;
boost::function<CUresult ( int*)> cuDriverGetVersion_;

boost::function<CUresult (int *)> cuDeviceGetCount_;
boost::function<CUresult ( CUdevice *, int)> cuDeviceGet_;
boost::function<CUresult ( char*, int, CUdevice)> cuDeviceGetName_;
boost::function<CUresult ( int*, int*, CUdevice)> cuDeviceComputeCapability_;
boost::function<CUresult ( size_t*, CUdevice)> cuDeviceTotalMem_;
boost::function<CUresult ( int*, CUdevice_attribute, CUdevice)> cuDeviceGetAttribute_;

boost::function<CUresult ( CUcontext*, unsigned int, CUdevice)> cuCtxCreate_;
boost::function<CUresult ( CUcontext)> cuCtxDestroy_;
boost::function<CUresult ( CUcontext)> cuCtxPushCurrent_;
boost::function<CUresult ( CUcontext*)> cuCtxPopCurrent_;
boost::function<CUresult ( CUcontext*)> cuCtxGetCurrent_;
boost::function<CUresult ( CUcontext*, unsigned int)> cuCtxAttach_;
boost::function<CUresult ( CUcontext)> cuCtxDetach_;

#ifdef BASE_WITH_GL
    boost::function<CUresult ( CUcontext*, unsigned int, CUdevice)> cuGLCtxCreate_;
    boost::function<CUresult ( CUgraphicsResource*, GLuint, unsigned int)> cuGraphicsGLRegisterBuffer_;
    boost::function<CUresult ( CUgraphicsResource*, GLuint, GLenum target, unsigned int)> cuGraphicsGLRegisterImage_;
    boost::function<CUresult ( unsigned int*, CUdevice*, unsigned int, CUGLDeviceList)> cuGLGetDevices_;
#endif

void reset_all_function_pointers()
{
	cuInit_ = 0;
	cuDriverGetVersion_ = 0;
	cuDeviceGetCount_ = 0;
	cuDeviceGet_ = 0;
	cuDeviceGetName_ = 0;
	cuDeviceGetAttribute_ = 0;
    cuDeviceComputeCapability_ = 0;
    cuDeviceTotalMem_ = 0;
	cuCtxCreate_ = 0;
	cuCtxDestroy_ = 0;
	cuCtxPushCurrent_ = 0;
	cuCtxPopCurrent_ = 0;
	cuCtxGetCurrent_ = 0;
	cuCtxAttach_ = 0;
	cuCtxDetach_ = 0;

    #ifdef BASE_WITH_GL
        cuGLCtxCreate_ = 0;
        cuGraphicsGLRegisterBuffer_ = 0;
        cuGraphicsGLRegisterImage_ = 0;
        cuGLGetDevices_ = 0;
    #endif
}

void *get_proc_ex( const std::string& name, bool required = true)
{
	BASE_ASSERT( cuda_lib_);

    #if defined __linux__ || defined __APPLE__
        void *fun = dlsym( cuda_lib_, name.c_str());
        return fun;
    #endif
}

void *get_proc_ex_v2( const std::string& name, bool required = true)
{
	BASE_ASSERT( cuda_lib_);

    #if defined __linux__ || defined __APPLE__
    	std::string name_v2 = name + std::string( "_v2");
        void *fun = dlsym( cuda_lib_, name_v2.c_str());
    	return fun;
    #endif
}

bool load_libcuda()
{
    // check if already loaded
    if( cuda_lib_)
        return true;

    #if defined __linux__ || defined __APPLE__
    	cuda_lib_ = dlopen( "libcuda.so", RTLD_NOW);
    #endif

    if( !cuda_lib_)
		return false;

	// init all function pointers here
	if( !( cuInit_ = reinterpret_cast<CUresult (*)( unsigned int)>( get_proc_ex( "cuInit"))))
		return false;

	if( !( cuDriverGetVersion_ = reinterpret_cast<CUresult (*)( int *)>( get_proc_ex( "cuDriverGetVersion"))))
		return false;

	if( !( cuDeviceGetCount_ = reinterpret_cast<CUresult (*)( int*)>( get_proc_ex( "cuDeviceGetCount"))))
		return false;

	if( !( cuDeviceGet_ = reinterpret_cast<CUresult (*)( CUdevice *, int)>( get_proc_ex( "cuDeviceGet"))))
		return false;

    if( !( cuDeviceGetName_ = reinterpret_cast<CUresult (*)( char*, int, CUdevice)>( get_proc_ex( "cuDeviceGetName"))))
        return false;

    if( !( cuDeviceComputeCapability_ = reinterpret_cast<CUresult (*)( int*, int*, CUdevice)>( get_proc_ex( "cuDeviceComputeCapability"))))
        return false;

    if( !( cuDeviceTotalMem_ = reinterpret_cast<CUresult (*)( size_t*, CUdevice)>( get_proc_ex_v2( "cuDeviceTotalMem"))))
        return false;

    if( !( cuDeviceGetAttribute_ = reinterpret_cast<CUresult (*)( int*, CUdevice_attribute, CUdevice)>( get_proc_ex( "cuDeviceGetAttribute"))))
        return false;

	if( !( cuCtxCreate_ = reinterpret_cast<CUresult (*)( CUcontext*, unsigned int, CUdevice)>( get_proc_ex_v2( "cuCtxCreate"))))
		return false;

	if( !( cuCtxDestroy_ = reinterpret_cast<CUresult (*)( CUcontext)>( get_proc_ex_v2( "cuCtxDestroy"))))
		return false;

	if( !( cuCtxPushCurrent_ = reinterpret_cast<CUresult (*)( CUcontext)>( get_proc_ex_v2( "cuCtxPushCurrent"))))
		return false;

	if( !( cuCtxPopCurrent_ = reinterpret_cast<CUresult (*)( CUcontext*)>( get_proc_ex_v2( "cuCtxPopCurrent"))))
		return false;

	if( !( cuCtxGetCurrent_ = reinterpret_cast<CUresult (*)( CUcontext*)>( get_proc_ex( "cuCtxGetCurrent"))))
		return false;

	if( !( cuCtxAttach_ = reinterpret_cast<CUresult (*)( CUcontext*, unsigned int)>( get_proc_ex( "cuCtxAttach"))))
		return false;

	if( !( cuCtxDetach_ = reinterpret_cast<CUresult (*)( CUcontext)>( get_proc_ex( "cuCtxDetach"))))
		return false;

    #ifdef BASE_WITH_GL

        if( !( cuGLCtxCreate_ = reinterpret_cast<CUresult (*)( CUcontext*, unsigned int, CUdevice)>( get_proc_ex( "cuGLCtxCreate"))))
            return false;

        if( !( cuGraphicsGLRegisterBuffer_ = reinterpret_cast<CUresult (*)( CUgraphicsResource*, GLuint, unsigned int)>( get_proc_ex( "cuGraphicsGLRegisterBuffer"))))
            return false;

        if( !( cuGraphicsGLRegisterImage_ = reinterpret_cast<CUresult (*)( CUgraphicsResource*, GLuint, GLenum target, unsigned int)>( get_proc_ex( "cuGraphicsGLRegisterImage"))))
            return false;

        if( !( cuGLGetDevices_ = reinterpret_cast<CUresult (*)( unsigned int*, CUdevice*, unsigned int, CUGLDeviceList)>( get_proc_ex( "cuGLGetDevices"))))
            return false;
    #endif

	return true;
}

} // unnamed

void unload_libcuda()
{
	if( cuda_lib_)
	{
		dlclose( cuda_lib_);
		cuda_lib_ = 0;
	}

	reset_all_function_pointers();
}

void check_cu_error( CUresult err)
{
	switch( err)
	{
		case CUDA_SUCCESS:
			return;

		case CUDA_ERROR_OUT_OF_MEMORY:
			throw out_of_memory();

		default:
			throw driver_error( err);
	};
}

void cu_init( unsigned int flags)
{
	bool result = load_libcuda();

	if( !result)
	{
		reset_all_function_pointers();
		throw std::runtime_error( "Couldn't dynamic load cuda library.");
    }

	BASE_ASSERT( cuInit_);
	check_cu_error( cuInit_( flags));
}

int cu_driver_get_version()
{
    BASE_ASSERT( cuDriverGetVersion_);
    int version = 0;
    cuDriverGetVersion_( &version);
    return version;
}

int cu_device_get_count()
{
    BASE_ASSERT( cuDeviceGetCount_);
    int num;
    check_cu_error( cuDeviceGetCount_( &num));
    return num;
}

CUdevice cu_device_get( int ordinal)
{
	BASE_ASSERT( cuDeviceGet_);
	CUdevice dev;
	check_cu_error( cuDeviceGet_( &dev, ordinal));
	return dev;
}

void cu_device_get_name( char *name, int len, CUdevice dev)
{
    BASE_ASSERT( cuDeviceGetName_);
    check_cu_error( cuDeviceGetName_( name, len, dev));
}

void cu_device_compute_capability( int *major, int *minor, CUdevice dev)
{
    BASE_ASSERT( cuDeviceComputeCapability_);
    check_cu_error( cuDeviceComputeCapability_( major, minor, dev));
}

size_t cu_device_total_mem( CUdevice dev)
{
    BASE_ASSERT( cuDeviceTotalMem_);
    size_t mem;
    check_cu_error( cuDeviceTotalMem_( &mem, dev));
    return mem;
}

int cu_device_get_attribute( CUdevice_attribute attrib, CUdevice dev)
{
    BASE_ASSERT( cuDeviceGetAttribute_);
    int pi;
    check_cu_error( cuDeviceGetAttribute_( &pi, attrib, dev));
    return pi;
}

void cu_ctx_create( CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	BASE_ASSERT( cuCtxCreate_);
	check_cu_error( cuCtxCreate_( pctx, flags, dev));
}

void cu_ctx_destroy( CUcontext ctx)
{
	BASE_ASSERT( cuCtxDestroy_);

	// in this case, we don't check for errors, as
	// this can be used in destructors, and throwing
    // inside a destructor is bad...
	cuCtxDestroy_( ctx);
}

void cu_ctx_push_current( CUcontext ctx)
{
	BASE_ASSERT( cuCtxPushCurrent_);
	check_cu_error( cuCtxPushCurrent_( ctx));
}

void cu_ctx_pop_current( CUcontext *pctx)
{
	BASE_ASSERT( cuCtxPopCurrent_);
	check_cu_error( cuCtxPopCurrent_( pctx));
}

void cu_ctx_get_current( CUcontext *pctx)
{
	BASE_ASSERT( cuCtxGetCurrent_);
	check_cu_error( cuCtxGetCurrent_( pctx));
}

void cu_ctx_attach( CUcontext *pctx, unsigned int flags)
{
    BASE_ASSERT( cuCtxAttach_);
    check_cu_error( cuCtxAttach_( pctx, flags));
}

void cu_ctx_detach( CUcontext ctx)
{
	BASE_ASSERT( cuCtxDetach_);
    check_cu_error( cuCtxDetach_( ctx));
}

#ifdef BASE_WITH_GL

    void cuGLCtxCreate( CUcontext *pCtx, unsigned int Flags, CUdevice device)
    {
        BASE_ASSERT( cuGLCtxCreate_);
        check_cu_error( cuGLCtxCreate_( pCtx, Flags, device));
    }

    void cuGraphicsGLRegisterBuffer( CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags)
    {
        BASE_ASSERT( cuGraphicsGLRegisterBuffer_);
        check_cu_error( cuGraphicsGLRegisterBuffer_( pCudaResource, buffer, Flags));
    }

    void cuGraphicsGLRegisterImage( CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags)
    {
        BASE_ASSERT( cuGraphicsGLRegisterImage_);
        check_cu_error( cuGraphicsGLRegisterImage_( pCudaResource, image, target, Flags));
    }

    void cuGLGetDevices( unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                            unsigned int cudaDeviceCount, CUGLDeviceList deviceList)
    {
        BASE_ASSERT( cuGLGetDevices_);
        check_cu_error( cuGLGetDevices_( pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList));
    }

#endif

} // namespace
} // namespace
