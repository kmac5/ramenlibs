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

#ifndef RAMEN_COLOR_RGB_TO_HSV_HPP
#define RAMEN_COLOR_RGB_TO_HSV_HPP

#include<ramen/color/config.hpp>

#include<algorithm>

#include<boost/utility/enable_if.hpp>
#include<boost/type_traits/is_floating_point.hpp>

#include<ramen/math/cmath.hpp>

#include<ramen/color/color3.hpp>

namespace ramen
{
namespace color
{

template<class T>
color3_t<T,hsv_t> rgb_to_hsv( const color3_t<T,rgb_t>& c,
                              typename boost::enable_if_c<boost::is_floating_point<T>::value>::type *unused = 0)
{
    color3_t<T,hsv_t> result;

    T r = c.x;
    T g = c.y;
    T b = c.z;

    // normalize RGB to 0 .. 1
    if( r < 0) r = T( 0);
    if( g < 0) g = T( 0);
    if( b < 0) b = T( 0);

    T scale = std::max( r, std::max( g, b));

    if( scale < T( 1.0))
        scale = T( 1.0);

    r /= scale;
    g /= scale;
    b /= scale;

    T maxv = std::max( r, std::max( g, b));
    T minv = std::min( r, std::min( g, b));

    result.x = T( 0);
    result.y = T( 0);
    result.z = maxv * scale;

    if( maxv != minv)
    {
        const T f = ( r == minv) ? ( g - b) : (( g == minv) ? ( b - r) : ( r - g)),
                i = ( r == minv) ? T( 3.0) : (( g == minv) ? T( 5.0) : T( 1.0));

        result.x = ( i - f / ( maxv - minv));

        if( result.x >= T( 6.0))
            result.x -= T( 6.0);

        result.x /= T( 6.0);
        result.y = ( maxv - minv) / maxv;
    }

    return result;
}

template<class T>
color3_t<T,rgb_t> hsv_to_rgb( const color3_t<T,hsv_t>& c,
                              typename boost::enable_if_c<boost::is_floating_point<T>::value>::type *unused = 0)
{
    T scale = std::max( c.z, T( 1.0));
    T h = c.x;
    T s = c.y;
    T v = c.z / scale;

    color3_t<T,rgb_t> result( T( 0));

    if( h == T( 0) && s == T( 0))
        result.x = result.y = result.z = v;
    else
    {
        h *= T( 6.0);
        const int i = static_cast<int>( math::cmath<T>::floor( h));
        const T f = ( i & 1) ? ( h - i) : ( T( 1.0) - h + i),
                m = v * ( T( 1.0) - s),
                n = v * ( T( 1.0) - s * f);

        switch( i)
        {
            case 6:
            case 0:
                result.x = v; result.y = n; result.z = m;
            break;

            case 1: result.x = n; result.y = v; result.z = m; break;
            case 2: result.x = m; result.y = v; result.z = n; break;
            case 3: result.x = m; result.y = n; result.z = v; break;
            case 4: result.x = n; result.y = m; result.z = v; break;
            case 5: result.x = v; result.y = m; result.z = n; break;
        }
    }

    result.x *= scale;
    result.y *= scale;
    result.z *= scale;
    return result;
}

} // color
} // ramen

#endif
