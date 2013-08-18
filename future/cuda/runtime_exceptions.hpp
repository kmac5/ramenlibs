//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#ifndef BASE_CUDA_RUNTIME_EXCEPTIONS_HPP
#define BASE_CUDA_RUNTIME_EXCEPTIONS_HPP

#include<base/config.hpp>

#include<base/cuda/exceptions.hpp>

#include<cuda_runtime_api.h>

namespace base
{
namespace cuda
{

class BASE_API runtime_error : public error
{
public:

	explicit runtime_error( cudaError_t err);

	cudaError_t cuda_error() const { return error_;}

protected:

	static const char *cuda_error_to_str( cudaError_t e);

	cudaError_t error_;
};

} // namespace
} // namespace

#endif
