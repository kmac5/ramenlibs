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

#include<ramen/geo/exceptions.hpp>

#include<stdexcept>

namespace ramen
{
namespace geo
{

attribute_not_found::attribute_not_found( const core::name_t& name)
{
    message_ = boost::move( core::make_string( "attribute not found: ", name.c_str()));
}

const char *attribute_not_found::what() const
{
    return message_.c_str();
}

bad_shape_type::bad_shape_type( shape_type_t type) : type_( type)
{
    message_ = boost::move( core::make_string( "bad shape type, type: ", shape_type_to_string( type)));
}

const char *bad_shape_type::what() const
{
    return message_.c_str();
}

bad_shape_cast::bad_shape_cast( shape_type_t src_type, shape_type_t dst_type) : src_type_( src_type),
                                                                                dst_type_( dst_type)
{
    message_ = boost::move( core::make_string( "bad shape cast: src_type = ", shape_type_to_string( src_type),
                                                              " dst_type = ", shape_type_to_string( dst_type)));
}

const char *bad_shape_cast::what() const
{
    return message_.c_str();
}

} // geo
} // ramen
