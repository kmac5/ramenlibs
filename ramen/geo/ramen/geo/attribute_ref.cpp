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

#include<ramen/geo/attribute_ref.hpp>

namespace ramen
{
namespace geo
{

const_attribute_ref_t::const_attribute_ref_t( const arrays::array_t *data)
{
    data_ = const_cast<arrays::array_t*>( data);
}

bool const_attribute_ref_t::valid() const
{
    return data_ != 0;
}

const arrays::array_t& const_attribute_ref_t::array() const
{
    assert( valid());

    return *data_;
}

core::type_t const_attribute_ref_t::array_type() const
{
    return array().type();
}

attribute_ref_t::attribute_ref_t( arrays::array_t *data) : const_attribute_ref_t( data) {}

arrays::array_t& attribute_ref_t::array()
{
    assert( valid());

    return *data_;
}

} // geo
} // ramen
