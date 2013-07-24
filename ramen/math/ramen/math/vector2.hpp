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

#ifndef RAMEN_MATH_VECTOR2_HPP
#define RAMEN_MATH_VECTOR2_HPP

#include<ramen/math/config.hpp>

#include<cassert>
#include<limits>

#include<boost/operators.hpp>

#ifdef RAMEN_WITH_HALF
    #include<ramen/core/half.hpp>
#endif

#include<ramen/math/cmath.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Two dimensional vector.
*/
template<class T>
class vector2_t     : boost::addable<vector2_t<T>
                    , boost::subtractable<vector2_t<T>
                    , boost::dividable2<vector2_t<T>, T
                    , boost::multipliable2<vector2_t<T>, T
                    , boost::equality_comparable<vector2_t<T>
                    > > > > >
{
public:

    typedef T value_type;

    static unsigned int	size()          { return 2;}
    static unsigned int	dimensions()    { return 2;}

    vector2_t() {}

    explicit vector2_t( T xx) : x( xx), y( xx) {}

    vector2_t( T xx, T yy) : x( xx), y( yy) {}

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

    vector2_t<T>& operator+=( const vector2_t<T>& vec)
    {
        x += vec.x;
        y += vec.y;
        return *this;
    }

    vector2_t<T>& operator-=( const vector2_t<T>& vec)
    {
        x -= vec.x;
        y -= vec.y;
        return *this;
    }

    vector2_t<T>& operator*=( T s)
    {
        x *= s;
        y *= s;
        return *this;
    }

    vector2_t<T>& operator/=( T s)
    {
        x /= s;
        y /= s;
        return *this;
    }

    vector2_t<T> operator-() const
    {
        return vector2_t<T>( -x, -y);
    }

    bool operator==( const vector2_t<T>& other) const
    {
        return x == other.x && y == other.y;
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
    }

    T x, y;
};

template<class T>
vector2_t<T> normalize( const vector2_t<T>& v)
{
    vector2_t<T> x( v);
    x.normalize();
    return x;
}

template<class T>
T dot( const vector2_t<T>& a, const vector2_t<T>& b)
{
    return ( a.x * b.x) + ( a.y * b.y);
}

template<class T>
T cross( const vector2_t<T>& a, const vector2_t<T>& b)
{
    return a.x * b.y - a.y * b.x;
}

// typedefs
typedef vector2_t<float>    vector2f_t;
typedef vector2_t<double>   vector2d_t;
typedef vector2_t<int>      vector2i_t;

#ifdef RAMEN_WITH_HALF
    typedef vector2_t<half> vector2h_t;
#endif

} // math
} // ramen

#endif
