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

#ifndef RAMEN_MATH_VECTOR3_HPP
#define RAMEN_MATH_VECTOR3_HPP

#include<ramen/math/config.hpp>

#include<limits>
#include<cassert>

#include<boost/operators.hpp>
#include<boost/mpl/bool.hpp>
#include<boost/mpl/int.hpp>

#ifdef RAMEN_WITH_HALF
    #include<ramen/core/half.hpp>
#endif

#include<ramen/math/cmath.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Three dimensional vector.
*/
template<class T>
class vector3_t     : boost::addable<vector3_t<T>
                    , boost::subtractable<vector3_t<T>
                    , boost::dividable2<vector3_t<T>, T
                    , boost::multipliable2<vector3_t<T>, T
                    , boost::equality_comparable<vector3_t<T>
                    > > > > >
{
public:

    typedef T                   value_type;
    typedef boost::mpl::int_<3> dimensions;
    typedef boost::mpl::int_<3> size;

    vector3_t() {}

    explicit vector3_t( T xx) : x( xx), y( xx), z( xx) {}

    vector3_t( T xx, T yy, T zz) : x( xx), y( yy), z( zz) {}

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

    // we need this for ramen. will be removed when possible
    T operator[]( unsigned int index) const
    {
        assert( index < size());

        return static_cast<const T*>( &x)[index];
    }

    // we need this for ramen. will be removed when possible
    T& operator[]( unsigned int index)
    {
        return (*this)( index);
    }

    // operators

    vector3_t<T>& operator+=( const vector3_t<T>& vec)
    {
        x += vec.x;
        y += vec.y;
        z += vec.z;
        return *this;
    }

    vector3_t<T>& operator-=( const vector3_t<T>& vec)
    {
        x -= vec.x;
        y -= vec.y;
        z -= vec.z;
        return *this;
    }

    vector3_t<T>& operator*=( T s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    vector3_t<T>& operator/=( T s)
    {
        assert( s != T(0));

        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    vector3_t<T> operator-() const
    {
        return vector3_t<T>( -x, -y, -z);
    }

    bool operator==( const vector3_t<T>& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    T length2() const
    {
        return dot( *this, *this);
    }

    T length() const
    {
        return cmath<T>::sqrt( length2());
    }

    void normalize()
    {
        T l = length();
        assert( l > T(0));

        x /= l;
        y /= l;
        z /= l;
    }

    T x, y, z;
};

template<class T>
vector3_t<T> normalize( const vector3_t<T>& v)
{
    vector3_t<T> x( v);
    x.normalize();
    return x;
}

template<class T>
T dot( const vector3_t<T>& a, const vector3_t<T>& b)
{
    return ( a.x * b.x) + ( a.y * b.y) + ( a.z * b.z);
}

template<class T>
vector3_t<T> cross( const vector3_t<T>& a, const vector3_t<T>& b)
{
    return vector3_t<T>( a.y * b.z - a.z * b.y,
                         a.z * b.x - a.x * b.z,
                         a.x * b.y - a.y * b.x);
}

typedef vector3_t<float>    vector3f_t;
typedef vector3_t<double>   vector3d_t;

#ifdef RAMEN_WITH_HALF
    typedef vector3_t<half> vector3h_t;
#endif

} // math
} // ramen

#endif
