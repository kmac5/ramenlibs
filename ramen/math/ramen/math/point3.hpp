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

#ifndef RAMEN_MATH_POINT3_HPP
#define RAMEN_MATH_POINT3_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/vector3.hpp>
#include<ramen/math/normal.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Three dimensional point.
*/
template<class T>
class point3_t  : boost::addable2<point3_t<T>, vector3_t<T>
                , boost::subtractable2<point3_t<T>, vector3_t<T>
                , boost::addable2<point3_t<T>, normal_t<T>
                , boost::subtractable2<point3_t<T>, normal_t<T>
                , boost::equality_comparable<point3_t<T>
                > > > > >
{
public:

    typedef T                           value_type;
    typedef boost::mpl::int_<3>         size_type;
    typedef boost::mpl::bool_<false>    is_homogeneus_type;
    typedef vector3_t<T>                vector_type;

    point3_t() {}

    explicit point3_t( T xx) : x( xx), y( xx), z( xx) {}

    point3_t( T xx, T yy, T zz) : x( xx), y( yy), z( zz) {}

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

    point3_t<T>& operator+=( const vector3_t<T>& vec)
    {
        x += vec.x;
        y += vec.y;
        z += vec.z;
        return *this;
    }

    point3_t<T>& operator-=( const vector3_t<T>& vec)
    {
        x -= vec.x;
        y -= vec.y;
        z -= vec.z;
        return *this;
    }

    point3_t<T>& operator+=( const normal_t<T>& n)
    {
        x += n.x;
        y += n.y;
        z += n.z;
        return *this;
    }

    point3_t<T>& operator-=( const normal_t<T>& n)
    {
        x -= n.x;
        y -= n.y;
        z -= n.z;
        return *this;
    }

    vector3_t<T> operator-( const point3_t<T>& other) const
    {
        return vector3_t<T>( x - other.x, y - other.y, z - other.z);
    }

    bool operator==( const point3_t<T>& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    static point3_t<T> origin()
    {
        return point3_t<T>( 0);
    }

    T x, y, z;
};

template<class T>
T squared_distance( const point3_t<T>& a, const point3_t<T>& b)
{
    T dx = b.x - a.x;
    T dy = b.y - a.y;
    T dz = b.z - a.z;
    return dx * dx + dy * dy + dz * dz;
}

template<class T>
T distance( const point3_t<T>& a, const point3_t<T>& b)
{
    return cmath<T>::sqrt( squared_distance( a, b));
}

// typedefs
typedef point3_t<float>     point3f_t;
typedef point3_t<double>    point3d_t;
typedef point3_t<int>       point3i_t;

#ifdef RAMEN_WITH_HALF
    typedef point3_t<half> point3h_t;
#endif

} // math
} // ramen

#endif
