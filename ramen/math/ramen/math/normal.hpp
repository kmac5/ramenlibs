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

#ifndef RAMEN_MATH_NORMAL_HPP
#define RAMEN_MATH_NORMAL_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/vector3.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Normal vector.
*/
template<class T>
class normal_t      : boost::addable<normal_t<T>
                    , boost::multipliable2<normal_t<T>, T
                    , boost::dividable2<normal_t<T>, T
                    , boost::equality_comparable<normal_t<T>
                    > > > >
{
public:

    typedef T                   value_type;
    typedef boost::mpl::int_<3> size_type;

    normal_t() {}

    explicit normal_t( T xx) : x( xx), y( xx), z( xx) {}

    normal_t( T xx, T yy, T zz) : x( xx), y( yy), z( zz) {}

    explicit normal_t( const vector3_t<T>& v) : x( v.x), y( v.y), z( v.z) {}

    // operators

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

    normal_t<T>& operator+=( const normal_t<T>& n)
    {
        x += n.x;
        y += n.y;
        return *this;
    }

    normal_t<T>& operator-=( const normal_t<T>& n)
    {
        x -= n.x;
        y -= n.y;
        return *this;
    }

    normal_t<T>& operator*=( T s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    normal_t<T>& operator/=( T s)
    {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // for regular concept
    bool operator==( const normal_t<T>& other) const
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
normal_t<T> normalize( const normal_t<T>& n)
{
    normal_t<T> x( n);
    x.normalize();
    return x;
}

template<class T>
T dot( const normal_t<T>& a, const vector3_t<T>& b)
{
    return ( a.x * b.x) + ( a.y * b.y) + ( a.z * b.z);
}

template<class T>
T dot( const vector3_t<T>& a, const normal_t<T>& b)
{
    return ( a.x * b.x) + ( a.y * b.y) + ( a.z * b.z);
}

template<class T>
T dot( const normal_t<T>& a, const normal_t<T>& b)
{
    return ( a.x * b.x) + ( a.y * b.y) + ( a.z * b.z);
}

typedef normal_t<float>     normalf_t;
typedef normal_t<double>    normald_t;

#ifdef RAMEN_WITH_HALF
    typedef normal_t<half> normalh_t;
#endif

} // math
} // ramen

#endif
