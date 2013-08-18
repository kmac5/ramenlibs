//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#ifndef BASE_CUDA_EXCEPTIONS_HPP
#define BASE_CUDA_EXCEPTIONS_HPP

#include<base/config.hpp>

#include<stdexcept>

#include<cuda.h>
#include<cuda_runtime_api.h>

namespace base
{
namespace cuda
{

class BASE_API error : public std::exception
{
public:
	
	explicit error( const std::string& msg);
    virtual ~error() throw();

    virtual const char* what() const throw();

private:

    std::string msg_;
};

struct BASE_API out_of_memory : std::bad_alloc {};

} // namespace
} // namespace

#endif
