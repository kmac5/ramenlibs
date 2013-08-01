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

#ifndef RAMEN_COLOR_COLOR3_HPP
#define RAMEN_COLOR_COLOR3_HPP

#include<ramen/color/config.hpp>

#include<cassert>
#include<algorithm>
#include<iostream>

#include<boost/static_assert.hpp>
#include<boost/operators.hpp>

#include<ramen/color/colorspace.hpp>

#ifdef RAMEN_WITH_HALF
    #include<ramen/core/half.hpp>
#endif

namespace ramen
{
namespace color
{

/*!
\brief A 3 component color.
*/
template<class T, class Colorspace = rgb_t>
class color3_t    : boost::addable<color3_t<T,Colorspace>
                  , boost::subtractable<color3_t<T,Colorspace>
                  , boost::dividable2<color3_t<T,Colorspace>, T
                  , boost::multipliable2<color3_t<T,Colorspace>, T
                  , boost::equality_comparable<color3_t<T,Colorspace>
                  > > > > >
{
public:

    typedef T                                       value_type;
    typedef Colorspace                              colorspace_type;
    typedef typename colorspace_type::size          size;

    BOOST_STATIC_ASSERT(( colorspace_type::size::value == 3));

    color3_t() {}

    explicit color3_t( T xx) : x( xx), y( xx), z( xx) {}

    color3_t( T xx, T yy, T zz) : x( xx), y( yy), z( zz) {}

    T operator()( unsigned int index) const
    {
        assert( index < size::value);

        return static_cast<const T*>( &x)[index];
    }

    T& operator()( unsigned int index)
    {
        assert( index < size::value);

        return static_cast<T*>( &x)[index];
    }

    color3_t<T,Colorspace>& operator+=( const color3_t<T,Colorspace>& c)
    {
        x += c.x;
        y += c.y;
        z += c.z;
        return *this;
    }

    color3_t<T,Colorspace>& operator-=( const color3_t<T,Colorspace>& c)
    {
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return *this;
    }

    color3_t<T,Colorspace>& operator*=( T s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    color3_t<T,Colorspace>& operator/=( T s)
    {
        assert( s != T(0));

        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    bool operator==( const color3_t<T,Colorspace>& other) const
    {
        return ( x == other.x && y == other.y && z == other.z);
    }

    void clamp( T lo = T(0), T hi = T(1))
    {
        x = std::max( lo, std::min( hi, x));
        y = std::max( lo, std::min( hi, y));
        z = std::max( lo, std::min( hi, z));
    }

    T min_component() const
    {
        return std::min( x, std::min( y, z));
    }

    T max_component() const
    {
        return std::max( x, std::max( y, z));
    }

    T x, y, z;
};

template<class T, class Colorspace>
color3_t<T,Colorspace> clamp( const color3_t<T,Colorspace>& c, T lo = T(0), T hi = T(1))
{
    color3_t<T> x( c);
    c.clamp( lo, hi);
    return x;
}

template<class T>
T luminance( const color3_t<T, rgb_t>& c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

template<class T, class Colorspace>
std::ostream& operator<<( std::ostream& os, const color3_t<T,Colorspace>& c)
{
    os << c.x << ", " << c.y << ", " << c.z;
    return os;
}

// typedefs
typedef color3_t<float,rgb_t>     color3f_t;
typedef color3_t<double,rgb_t>    color3d_t;

#ifdef RAMEN_WITH_HALF
    typedef color3_t<half,rgb_t> color3h_t;
#endif

} // color
} // ramen

#endif
