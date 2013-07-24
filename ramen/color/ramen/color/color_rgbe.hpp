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

#ifndef RAMEN_COLOR_COLOR_RGBE_HPP
#define RAMEN_COLOR_COLOR_RGBE_HPP

#include<ramen/color/config.hpp>

#include<algorithm>

#include<boost/integer.hpp>
#include<boost/static_assert.hpp>
#include<boost/type_traits/is_floating_point.hpp>

#include<ramen/math/cmath.hpp>

#include<ramen/color/color3.hpp>

namespace ramen
{
namespace color
{

/*!
\brief A 3 component rgbe compressed color.
*/
class color_rgbe_t : boost::equality_comparable<color_rgbe_t>
{
public:

    typedef boost::uint8_t value_type;

    color_rgbe_t() {}

    template<class T>
    explicit color_rgbe_t( const color3_t<T>& col)
    {
        BOOST_STATIC_ASSERT(( boost::is_floating_point<T>::value));

        T v = std::max( std::max( col.x, col.y), col.z);
        if( v >= T( 1e-32f))
        {
            int e;
            v = math::cmath<T>::frexp( v, e) * T( 256.0) / v;
            r_ = static_cast<unsigned char>( col.x * v);
            g_ = static_cast<unsigned char>( col.y * v);
            b_ = static_cast<unsigned char>( col.z * v);
            e_ = static_cast<unsigned char>( e + 128);
        }
        else
            r_ = g_ = b_ = e_ = 0;
    }

    template<class T>
    operator color3_t<T>()
    {
        BOOST_STATIC_ASSERT(( boost::is_floating_point<T>::value));

        if( e_)
        {
            T f = math::cmath<T>::ldexp( T( 1.0), e_ - static_cast<int>( 128 + 8));
            return color3_t<T>( r_ * f, g_ * f, b_ * f);
        }

        return color3_t<T>( 0.0);
    }

    inline bool operator==( const color_rgbe_t& other) const
    {
        return r_ == other.r_ && g_ == other.g_ &&
               b_ == other.b_ && e_ == other.e_;
    }

private:

    boost::uint8_t r_, g_, b_, e_;
};

} // color
} // ramen

#endif