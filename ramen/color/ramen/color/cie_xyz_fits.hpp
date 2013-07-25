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

/*
    Based on:

    Simple Analytic Approximations to the CIE XYZ Color Matching Functions
    Chris Wyman, Peter-Pike Sloan and Peter Shirley
*/

#ifndef RAMEN_COLOR_CIE_XYZ_FITS_HPP
#define RAMEN_COLOR_CIE_XYZ_FITS_HPP

#include<ramen/color/config.hpp>

#include<ramen/math/cmath.hpp>

namespace ramen
{
namespace color
{

template<class T>
struct single_gaussian_1931_t
{
    typedef T value_type;

    T X( T wavelength) const
    {
        T a1 = T( 1.065);
        T b1 = T( 595.8);
        T c1 = T( 33.33);
        T tmp = ( wavelength - b1) / c1;
        T big_lobe = a1 * math::cmath<T>::exp( T( -0.5) * tmp * tmp);

        T a2 = T( 0.3660);
        T b2 = T( 446.8);
        T c2 = T( 19.44);
        tmp = ( wavelength - b2) / c2;
        T small_lobe = a2 * math::cmath<T>::exp( T( -0.5) * tmp * tmp);

        return small_lobe + big_lobe;
    }

    T Y( T wavelength) const
    {
        T a = T( 1.014);
        T b = T( 556.3);
        T c = T( 0.075);
        T tmp = ( math::cmath<T>::log( wavelength) - math::cmath<T>::log( b)) / c;
        return a * math::cmath<T>::exp( T( -0.5) * tmp * tmp);
    }

    T Z( T wavelength) const
    {
        T a = T( 1.839);
        T b = T( 449.8);
        T c = T( 0.051);
        T tmp = ( math::cmath<T>::log( wavelength) - math::cmath<T>::log( b)) / c;
        return a * math::cmath<T>::exp( T( -0.5) * tmp * tmp);
    }
};

template<class T>
struct single_gaussian_1964_t
{
    typedef T value_type;

    T X( T wavelength)
    {
        T val[4] = { T( 0.4), T( 1014.0), T( -0.02), T( -570.0)};
        T small_lobe = val[0] * math::cmath<T>::exp( -1250 * sqr( math::cmath<T>::log(( wavelength - val[3]) / val[1])));

        T val2[4] = { T( 1.13), T( 234.0), T( -0.001345), T( -1.799)};
        T big_lobe = val2[0] * math::cmath<T>::exp( -val2[1] * sqr( math::cmath<T>::log(( 1338 - wavelength) / T( 743.5))));

        return small_lobe + big_lobe;
    }

    T Y( T wavelength)
    {
        T val[4] = { T( 1.011), T( 556.1), T( 46.14), T( 0) };
        return val[0] * math::cmath<T>::exp( T( -0.5) * sqr(( wavelength-val[1]) / val[2])) + val[3];
    }

    T Z( T wavelength)
    {
        T val[4] = { T( 2.06), T( 180.4), T( 0.125), T( 266.0)};
        return val[0] * math::cmath<T>::exp( T( -32.0) * sqr( math::cmath<T>::log(( wavelength-val[3]) / val[1])));
    }

private:

    inline T sqr( T x) const
    {
        return x * x;
    }
};

template<class T>
struct multi_gaussian_1931_t
{
    typedef T value_type;

    T X( T wavelength)
    {
        T a1 = T( 1.065);
        T b1 = T( 595.8);
        T c1 = T( 33.33);
        T tmp = ( wavelength - b1) / c1;
        T big_lobe = a1 * math::cmath<T>::exp( T( -0.5) * tmp * tmp);

        T a2 = T( 0.3660);
        T b2 = T( 446.8);
        T c2 = T( 19.44);
        tmp = ( wavelength - b2) / c2;
        T small_lobe = a2 * math::cmath<T>::exp( T( -0.5) * tmp * tmp);

        return small_lobe + big_lobe;
    }

    T Y( T wavelength)
    {
        T a = T( 1.014);
        T b = T( 556.3);
        T c = T( 0.075);
        T tmp = ( math::cmath<T>::log( wavelength) - math::cmath<T>::log( b)) / c;
        return a * math::cmath<T>::exp( T( -0.5) * tmp * tmp);
    }

    T Z( T wavelength)
    {
        T a = T( 1.839);
        T b = T( 449.8);
        T c = T( 0.051);
        T tmp = ( math::cmath<T>::log( wavelength) - math::cmath<T>::log( b)) / c;
        return a * math::cmath<T>::exp( T( -0.5) * tmp * tmp);
    }
};

} // color
} // ramen

#endif
