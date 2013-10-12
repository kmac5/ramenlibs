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

#ifndef RAMEN_COLOR_RGBA_COLOR_HPP
#define RAMEN_COLOR_RGBA_COLOR_HPP

#include<ramen/color/config.hpp>

#include<ramen/color/color3.hpp>

namespace ramen
{
namespace color
{

/*!
\brief A RGBA color.
*/
template<class T>
class rgba_color_t : boost::equality_comparable<rgba_color_t<T> >
{
public:

    typedef T                                       value_type;
    typedef rgba_t                                  colorspace_type;
    typedef typename colorspace_type::size_type     size_type;

    rgba_color_t() {}

    rgba_color_t( T rr, T gg, T bb, T aa = T(1)) : r( rr), g( gg), b( bb), a( aa) {}
    
    rgba_color_t( T x, T alpha = T(1)) : r( x), g( x), b( x), a( alpha) {}
    
    explicit rgba_color_t( const color3_t<T, rgb_t>& c) : r( c.x), g( c.y), b( c.z), a( T(1))
    {
    }
    
    T operator()( unsigned int index) const
    {
        assert( index < size_type::value);

        return static_cast<const T*>( &r)[index];
    }

    T& operator()( unsigned int index)
    {
        assert( index < size_type::value);

        return static_cast<T*>( &r)[index];
    }

    // we need this for ramen. will be removed when possible
    T operator[]( unsigned int index) const
    {
        return (*this)( index);
    }

    // we need this for ramen. will be removed when possible
    T& operator[]( unsigned int index)
    {
        return (*this)( index);
    }

    // for regular concept
    bool operator==( const rgba_color_t<T>& other) const
    {
        return  r == other.r &&
                g == other.g &&
                b == other.b &&
                a == other.a;
    }

    void premultiply()
    {
        r *= a;
        g *= a;
        b *= a;
    }

    void unpremultiply()
    {
        if( a != T( 0))
        {
            r /= a;
            g /= a;
            b /= a;
        }
    }
    
    T r, g, b, a;
};

template<class T>
rgba_color_t<T> premultiply( const rgba_color_t<T>& c)
{
    rgba_color_t<T> x( c);
    x.premultiply();
    return x;
}

template<class T>
rgba_color_t<T> unpremultiply( const rgba_color_t<T>& c)
{
    rgba_color_t<T> x( c);
    x.unpremultiply();
    return x;
}
template<class T>
rgba_color_t<T> composite_over( const rgba_color_t<T>& a,
                                const rgba_color_t<T>& b)
{
    T one_minus_aa = T(1) - a.a;
    return rgba_color_t<T>( a.r + b.r * one_minus_aa,
                            a.g + b.g * one_minus_aa,
                            a.b + b.b * one_minus_aa,
                            a.a + b.a * one_minus_aa);
}

// typedefs
typedef rgba_color_t<boost::uint8_t>    rgba_colorc_t;
typedef rgba_color_t<boost::uint16_t>   rgba_colors_t;
typedef rgba_color_t<float>             rgba_colorf_t;
typedef rgba_color_t<double>            rgba_colord_t;

#ifdef RAMEN_WITH_HALF
    typedef rgba_color_t<half> rgba_colorh_t;
#endif

} // color
} // ramen

#endif
