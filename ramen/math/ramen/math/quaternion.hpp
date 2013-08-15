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

#ifndef RAMEN_MATH_QUATERNION_HPP
#define RAMEN_MATH_QUATERNION_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/matrix44.hpp>

#ifdef RAMEN_WITH_HALF
    #include<ramen/core/half.hpp>
#endif

namespace ramen
{
namespace math
{

/*!
\brief Quaternion class.
*/
template<class T>
class quaternion_t : boost::addable<quaternion_t<T>,
                     boost::subtractable<quaternion_t<T>,
                     boost::multipliable<quaternion_t<T>,
                     boost::multipliable2<quaternion_t<T>, T,
                     boost::dividable2<quaternion_t<T>, T > > > > >
{
public:

    quaternion_t() {}

    quaternion_t( T ss, T xx, T yy, T zz) : s( ss), x( xx), y( yy), z( zz)
    {
    }

    quaternion_t( vector3_t<T> axis, T ss = T(0)) : s( ss), x( axis.x), y( axis.y), z( axis.z)
    {
    }

    // operators
    quaternion_t<T>& operator+=( const quaternion_t<T>& q)
    {
        s += q.s;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    quaternion_t<T>& operator-=( const quaternion_t<T>& q)
    {
        s -= q.s;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    quaternion_t<T>& operator*=( const quaternion_t<T>& q)
    {
        T ss = s;
        T xx = x;
        T yy = y;
        T zz = z;
        s = ss * q.s - xx * q.x - yy * q.y - zz * q.z;
        x = ss * q.x + xx * q.s + yy * q.z - zz * q.y;
        y = ss * q.y - xx * q.z + yy * q.s + zz * q.x;
        z = ss * q.z + xx * q.y - yy * q.x + zz * q.s;
        return *this;
    }

    quaternion_t<T>& operator*=( T f)
    {
        s *= f;
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    quaternion_t<T>& operator/=( T f)
    {
        assert( f > T(0));

        s /= f;
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    T squared_norm() const
    {
        return s * s + x * x + y * y + z * z;
    }

    T norm() const
    {
        return cmath<T>::sqrt( squared_norm());
    }

    void normalize()
    {
        T l = norm();
        assert( l > T(0));

        s /= l;
        x /= l;
        y /= l;
        z /= l;
    }

    void conjugate()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    void inverse()
    {
        conjugate();
        *this /= squared_norm();
    }

    matrix44_t<T> to_matrix() const
    {
        matrix44_t<T> result;
        result( 0, 0) = 1 - 2 * ( y * y + z * z);
        result( 0, 1) = 2 * ( x * y + z * s);
        result( 0, 2) = 2 * ( z * x - y * s);
        result( 0, 3) = 0;

        result( 1, 0) = 2 * ( x * y - z * s);
        result( 1, 1) = 1 - 2 * ( z * z + x * x);
        result( 1, 2) = 2 * ( y * z + x * s);
        result( 1, 3) = 0;

        result( 2, 0) = 2 * ( z * x + y * s);
        result( 2, 1) = 2 * ( y * z - x * s);
        result( 2, 2) = 1 - 2 * ( y * y + x * x);
        result( 2, 3) = 0;

        result( 3, 0) = 0;
        result( 3, 1) = 0;
        result( 3, 2) = 0;
        result( 3, 3) = 1;
        return result;
    }

    T s, x, y, z;
};

template<class T>
quaternion_t<T> normalize( const quaternion_t<T>& q)
{
    quaternion_t<T> x( q);
    x.normalize();
    return x;
}

template<class T>
quaternion_t<T> conjugate( const quaternion_t<T>& q)
{
    quaternion_t<T> x( q);
    x.conjugate();
    return x;
}

template<class T>
quaternion_t<T> inverse( const quaternion_t<T>& q)
{
    quaternion_t<T> x( q);
    x.inverse();
    return x;
}

typedef quaternion_t<float>    quaternionf_t;
typedef quaternion_t<double>   quaterniond_t;

#ifdef RAMEN_WITH_HALF
    typedef quaternion_t<half> quaternionh_t;
#endif

} // math
} // ramen

#endif
