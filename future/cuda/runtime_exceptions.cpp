//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#include<base/cuda/runtime_exceptions.hpp>

namespace base
{
namespace cuda
{

runtime_error::runtime_error( cudaError_t err) : error( cuda_error_to_str( err)), error_( err) {}

const char *runtime_error::cuda_error_to_str( cudaError_t e)
{
	switch( e)
	{
		case cudaSuccess:
    		return "success";

		case cudaErrorMemoryAllocation:
			return "out of memory";

		default:
			return "unknown";
	};
}

} // namespace
} // namespace
