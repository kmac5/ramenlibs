//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#ifndef BASE_CUDA_DRIVER_EXCEPTIONS_HPP
#define BASE_CUDA_DRIVER_EXCEPTIONS_HPP

#include<base/config.hpp>

#include<base/cuda/exceptions.hpp>

#include<cuda.h>

namespace base
{
namespace cuda
{

class BASE_API driver_error : public error
{
public:

	explicit driver_error( CUresult err);

	CUresult cu_result() const { return error_;}

protected:

	static const char *curesult_to_str( CUresult e);

	CUresult error_;
};

} // namespace
} // namespace

#endif
