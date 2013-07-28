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

#ifndef RAMEN_GEO_EXCEPTIONS_HPP
#define RAMEN_GEO_EXCEPTIONS_HPP

#include<ramen/geo/config.hpp>

#include<ramen/core/exceptions.hpp>

#include<ramen/geo/shape_models/shape_types.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API attribute_not_found : public core::exception
{
public:

    explicit attribute_not_found( const core::name_t& name);

    virtual const char *what() const;

private:

    core::string8_t message_;
};

class RAMEN_GEO_API bad_shape_type : public core::exception
{
public:

    explicit bad_shape_type( shape_type_t type);

    virtual const char *what() const;

private:

    shape_type_t type_;
    core::string8_t message_;
};

class RAMEN_GEO_API bad_shape_cast : public core::exception
{
public:

    bad_shape_cast( shape_type_t src_type, shape_type_t dst_type);

    virtual const char *what() const;

private:

    shape_type_t src_type_, dst_type_;
    core::string8_t message_;
};

} // geo
} // ramen

#endif
