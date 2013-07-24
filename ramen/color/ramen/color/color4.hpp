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

#ifndef RAMEN_COLOR_COLOR4_HPP
#define RAMEN_COLOR_COLOR4_HPP

#include<ramen/color/config.hpp>

#include<cassert>

#include<boost/operators.hpp>

#ifdef RAMEN_WITH_HALF
    #include<ramen/core/half.hpp>
#endif

namespace ramen
{
namespace color
{

/*!
\brief A RGBA color.
*/
template<class T>
class color4_t : boost::equality_comparable<color4_t<T> >
{
public:

    typedef T value_type;

    static unsigned int	dimensions() { return 4;}

    color4_t() {}

    explicit color4_t( T x) : r( x), g( x), b( x), a( x) {}

    color4_t( T rr, T gg, T bb, T aa = T(1)) : r( rr), g( gg), b( bb), a( aa) {}

    T operator()( unsigned int index) const
    {
        assert( index < dimensions());

        return static_cast<const T*>( &r)[index];
    }

    T& operator()( unsigned int index)
    {
        assert( index < dimensions());

        return static_cast<T*>( &r)[index];
    }

    // for regular concept
    bool operator==( const color4_t<T>& other) const
    {
        return  r == other.r &&
                g == other.g &&
                b == other.b &&
                a == other.a;
    }

    T r, g, b, a;
};

// typedefs
typedef color4_t<float>     color4f_t;
typedef color4_t<double>    color4d_t;

#ifdef RAMEN_WITH_HALF
    typedef color4_t<half> color4h_t;
#endif

} // color
} // ramen

#endif
