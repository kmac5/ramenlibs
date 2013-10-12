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

#include<ramen/deep/pixel.hpp>

#include<ramen/core/global_names.hpp>

#include<ramen/arrays/array.hpp>

namespace ramen
{
namespace deep
{

pixel_t::pixel_t()
{
}

pixel_t::pixel_t( const pixel_t& other) : data_( other.data_)
{
}

void pixel_t::swap( pixel_t& other)
{
    std::swap( data_, other.data_);
}

const arrays::named_array_map_t& pixel_t::data() const
{    
    return data_;
}

arrays::named_array_map_t& pixel_t::data()
{
    return data_;
}

} // deep
} // ramen
