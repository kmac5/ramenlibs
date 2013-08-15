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

#include<ramen/core/exceptions.hpp>

namespace ramen
{
namespace core
{

runtime_error::runtime_error( string8_t message) : message_( boost::move( message)) {}

const char *runtime_error::what() const
{
    return message_.c_str();
}

bad_cast::bad_cast( const std::type_info& src_type, const std::type_info& dst_type)
{
    // TODO: do something with this...
    /*
    assert( src_type_ != dst_type_);
    message_ = boost::move( make_string( "bad cast: src_type = ", type_to_string( src_type),
                                                    " dst_type = ", type_to_string( dst_type)));
    */
    message_ = "bad cast";
}

const char *bad_cast::what() const
{
    return message_.c_str();
}

bad_type_cast::bad_type_cast( const type_t& src_type, const type_t& dst_type)
{
    assert( src_type != dst_type);
    message_ = boost::move( make_string( "bad cast: src_type = ", type_to_string( src_type),
                                                  " dst_type = ", type_to_string( dst_type)));
}

const char *bad_type_cast::what() const
{
    return message_.c_str();
}

not_implemented::not_implemented() {}

const char *not_implemented::what() const
{
    return "not implemented";
}

} // core
} // ramen
