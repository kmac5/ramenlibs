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

#ifndef RAMEN_MATH_HPOINT3_HPP
#define RAMEN_MATH_HPOINT3_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/point3.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Homogeneous three dimensional point.
*/
template<class T>
class hpoint3_t : boost::equality_comparable<hpoint3_t<T> >
{
public:

    typedef T                           value_type;
    typedef boost::mpl::int_<3>         dimensions;
    typedef boost::mpl::int_<4>         size;
    typedef boost::mpl::bool_<true>     is_homogeneus;

    hpoint3_t() {}

    explicit hpoint3_t( T xx) : x( xx), y( xx), z( xx), w( 1) {}

    hpoint3_t( T xx, T yy, T zz, T ww = T(1)) : x( xx), y( yy), z( zz), w( ww) {}

    explicit hpoint3_t( const point3_t<T>& p) : x( p.x), y( p.y), z( p.z), w( T(1)) {}

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

    bool operator==( const hpoint3_t<T>& other) const
    {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }

    T x, y, z, w;
};

template<class T>
point3_t<T> project( const hpoint3_t<T>& p)
{
    assert( p.w != T(0));

    return point3_t<T>( p.x / p.w, p.y / p.w, p.z / p.w);
}

// typedefs
typedef hpoint3_t<float>     hpoint3f_t;
typedef hpoint3_t<double>    hpoint3d_t;

#ifdef RAMEN_WITH_HALF
    typedef hpoint3_t<half> hpoint3h_t;
#endif

} // math
} // ramen

#endif
