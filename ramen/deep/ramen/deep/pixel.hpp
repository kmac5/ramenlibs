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

#ifndef RAMEN_DEEP_PIXEL_HPP
#define RAMEN_DEEP_PIXEL_HPP

#include<ramen/deep/config.hpp>

#include<boost/move/move.hpp>

#include<ramen/color/color3.hpp>

#include<ramen/arrays/named_array_map.hpp>
#include<ramen/arrays/array_ref.hpp>

namespace ramen
{
namespace deep
{

class RAMEN_DEEP_API pixel_t
{
    BOOST_COPYABLE_AND_MOVABLE( pixel_t)
    
public:
    
    explicit pixel_t( core::type_t alpha_type);
    ~pixel_t();

    // Copy constructor
    pixel_t( const pixel_t& other);

    // Copy assignment
    pixel_t& operator=( BOOST_COPY_ASSIGN_REF( pixel_t) other)
    {
        pixel_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    pixel_t( BOOST_RV_REF( pixel_t) other)
    {
        swap( other);
    }

    // Move assignment
    pixel_t& operator=( BOOST_RV_REF( pixel_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( pixel_t& other);
    
private:

    arrays::named_array_map_t data_;
    arrays::array_ref_t<float> *z_data_ref_;
};

inline void swap( pixel_t& x, pixel_t& y)
{
    x.swap( y);
}

} // deep
} // ramen

#endif
