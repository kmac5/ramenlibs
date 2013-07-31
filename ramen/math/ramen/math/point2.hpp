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

#ifndef RAMEN_MATH_POINT2_HPP
#define RAMEN_MATH_POINT2_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/vector2.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Two dimensional point.
*/
template<class T>
class point2_t  : boost::addable2<point2_t<T>, vector2_t<T>
                , boost::subtractable2<point2_t<T>, vector2_t<T>
                , boost::equality_comparable<point2_t<T>
                > > >
{
public:

    typedef T value_type;

    static const unsigned int dimensions    = 2;
    static const bool is_homogeneus         = false;
    static unsigned int	size()              { return 2;}

    point2_t() {}

    explicit point2_t( T xx) : x( xx), y( xx) {}

    point2_t( T xx, T yy) : x( xx), y( yy) {}

    T operator()( unsigned int index) const
    {
        assert( index < size());

        return static_cast<const T*>( &x)[index];
    }

    T& operator()( unsigned int index)
    {
        assert( index < size());

        return static_cast<T*>( &x)[index];
    }

    // operators

    point2_t<T>& operator+=( const vector2_t<T>& vec)
    {
        x += vec.x;
        y += vec.y;
        return *this;
    }

    point2_t<T>& operator-=( const vector2_t<T>& vec)
    {
        x -= vec.x;
        y -= vec.y;
        return *this;
    }

    vector2_t<T> operator-( const point2_t<T>& other) const
    {
        return vector2_t<T>( x - other.x, y - other.y);
    }

    bool operator==( const point2_t<T>& other) const
    {
        return x == other.x && y == other.y;
    }

    static point2_t<T> origin()
    {
        return point2_t<T>( 0);
    }

    T x, y;
};

template<class T>
T squared_distance( const point2_t<T>& a, const point2_t<T>& b)
{
    T dx = b.x - a.x;
    T dy = b.y - a.y;
    return dx * dx + dy * dy;
}

template<class T>
T distance( const point2_t<T>& a, const point2_t<T>& b)
{
    return cmath<T>::sqrt( squared_distance( a, b));
}

// typedefs
typedef point2_t<int>       point2i_t;
typedef point2_t<float>     point2f_t;
typedef point2_t<double>    point2d_t;

#ifdef RAMEN_WITH_HALF
    typedef point2_t<half> point2h_t;
#endif

} // math
} // ramen

#endif
