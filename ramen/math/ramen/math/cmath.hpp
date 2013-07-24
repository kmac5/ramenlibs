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

#ifndef RAMEN_MATH_CMATH_HPP
#define RAMEN_MATH_CMATH_HPP

#include<ramen/math/config.hpp>

#include<math.h>

namespace ramen
{
namespace math
{

template<class T>
struct cmath
{
    static T sqrt( T x)             { return ::sqrt( static_cast<double>( x));}
    static T fabs( T x)             { return ::fabs( static_cast<double>( x));}

    static T sin( T x)              { return ::sin( static_cast<double>( x));}
    static T cos( T x)              { return ::cos( static_cast<double>( x));}
    static T tan( T x)              { return ::tan( static_cast<double>( x));}

    static T exp( T x)              { return ::exp( static_cast<double>( x));}
    static T atan2( T x, T y)       { return ::atan2( static_cast<double>( x), static_cast<double>( y));}

    static T floor( T x)            { return ::floor( x);}
    static T ceil( T x)             { return ::ceil( x);}

    static T pow( T x, T y)         { return ::pow( x, y);}

    static T log10( T x)            { return ::log10( x);}

    static T ldexp( T x, int y)     { return ::ldexp( x, y);}
    static T frexp( T x, int& y)    { return ::frexp( x, &y);}
};

// float specialization.
template<>
struct cmath<float>
{
    static float sqrt( float x)             { return ::sqrtf( x);}
    static float fabs( float x)             { return ::fabsf( x);}

    static float sin( float x)              { return ::sinf( x);}
    static float cos( float x)              { return ::cosf( x);}
    static float tan( float x)              { return ::tanf( x);}

    static float exp( float x)              { return ::expf( x);}
    static float atan2( float x, float y)   { return ::atan2f( x, y);}

    static float floor( float x)            { return ::floorf( x);}
    static float ceil( float x)             { return ::ceilf( x);}

    static float pow( float x, float y)     { return ::powf( x, y);}

    static float log10( float x)            { return ::log10f( x);}

    static float ldexp( float x, int y)     { return ::ldexpf( x, y);}
    static float frexp( float x, int& y)    { return ::frexpf( x, &y);}
};

} // math
} // ramen

#endif
