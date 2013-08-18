//  (C) Copyright Esteban Tovagliari 2011.
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#include<base/cuda/exceptions.hpp>

namespace base
{
namespace cuda
{
	
error::error( const std::string& msg) : msg_( msg) {}

error::~error() throw() {}

const char *error::what() const throw() { return msg_.c_str();}

} // namespace
} // namespace
