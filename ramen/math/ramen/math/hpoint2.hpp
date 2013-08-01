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

#ifndef RAMEN_MATH_HPOINT2_HPP
#define RAMEN_MATH_HPOINT2_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/point2.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Homogeneus two dimensional point.
*/
template<class T>
class hpoint2_t : boost::equality_comparable<hpoint2_t<T> >
{
public:

    typedef T                           value_type;
    typedef boost::mpl::int_<3>         size_type;
    typedef boost::mpl::bool_<true>     is_homogeneus_type;

    hpoint2_t() {}

    explicit hpoint2_t( T xx) : x( xx), y( xx), w( 1) {}

    hpoint2_t( T xx, T yy, T ww = T(1)) : x( xx), y( yy), w( ww) {}

    explicit hpoint2_t( const point2_t<T>& p) : x( p.x), y( p.y), w( T(1)) {}

    T operator()( unsigned int index) const
    {
        assert( index < size_type::value);

        return static_cast<const T*>( &x)[index];
    }

    T& operator()( unsigned int index)
    {
        assert( index < size_type::value);

        return static_cast<T*>( &x)[index];
    }

    bool operator==( const hpoint2_t<T>& other) const
    {
        return x == other.x && y == other.y && w == other.w;
    }

    T x, y, w;
};

template<class T>
point2_t<T> project( const hpoint2_t<T>& p)
{
    assert( p.w != T(0));

    return point2_t<T>( p.x / p.w, p.y / p.w);
}

// typedefs
typedef hpoint2_t<float>     hpoint2f_t;
typedef hpoint2_t<double>    hpoint2d_t;

#ifdef RAMEN_WITH_HALF
    typedef hpoint2_t<half> hpoint2h_t;
#endif

} // math
} // ramen

#endif
