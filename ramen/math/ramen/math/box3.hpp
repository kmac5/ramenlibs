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

#ifndef RAMEN_MATH_BOX3_HPP
#define RAMEN_MATH_BOX3_HPP

#include<ramen/math/config.hpp>

#include<algorithm>
#include<limits>

#include<boost/operators.hpp>

#include<ramen/math/point3.hpp>
#include<ramen/math/matrix44.hpp>

namespace ramen
{
namespace math
{

/*!
\brief A three dimensional bounding box.
*/
template<class T>
class box3_t : boost::equality_comparable<box3_t<T> >
{
public:

    typedef T               value_type;
    typedef point3_t<T>     point_type;
    typedef vector3_t<T>    vector_type;

    box3_t()
    {
        reset();
    }

    box3_t( const point_type& p) : min( p), max( p) {}

    box3_t( const point_type& pmin, const point_type& pmax) : min( pmin), max( pmax) {}

    void reset()
    {
        min.x = min.y = min.z =  std::numeric_limits<T>::max();
        max.x = max.y = max.z = -std::numeric_limits<T>::max();
    }

    bool is_empty() const
    {
        return max.x < min.x || max.y < min.y || max.z < min.z;
    }

    vector_type size() const
    {
        return max - min;
    }

    point_type center() const
    {
        return min + ( size() * T(0.5));
    }

    void extend_by( const point_type& p)
    {
        min.x = std::min( min.x, p.x);
        max.x = std::max( max.x, p.x);
        min.y = std::min( min.y, p.y);
        max.y = std::max( max.y, p.y);
        min.z = std::min( min.z, p.z);
        max.z = std::max( max.z, p.z);
    }

    void extend_by( const box3_t<T>& b)
    {
        min.x = std::min( min.x, b.min.x);
        max.x = std::max( max.x, b.max.x);
        min.y = std::min( min.y, b.min.y);
        max.y = std::max( max.y, b.max.y);
        min.z = std::min( min.z, b.min.z);
        max.z = std::max( max.z, b.max.z);
    }

    void transform( const matrix44_t<T>& m)
    {
        // TODO: this could be optimized if m is affine
        box3_t<T> saved( *this);
        extend_by( saved.min * m);
        extend_by( point_type( saved.max.x, saved.min.y, saved.min.z) * m);
        extend_by( point_type( saved.max.x, saved.max.y, saved.min.z) * m);
        extend_by( point_type( saved.min.x, saved.max.y, saved.min.z) * m);
        extend_by( point_type( saved.max.x, saved.min.y, saved.max.z) * m);
        extend_by( point_type( saved.max.x, saved.max.y, saved.max.z) * m);
        extend_by( point_type( saved.min.x, saved.max.y, saved.max.z) * m);
        extend_by( saved.max * m);
    }

    void offset_by( const vector3_t<T>& v)
    {
        min += v;
        max += v;
    }

    bool contains( const point_type& p) const
    {
        if( p.x < min.x || p.x > max.x)
            return false;

        if( p.y < min.y || p.y > max.y)
            return false;

        if( p.z < min.z || p.z > max.z)
            return false;

        return true;
    }

    bool contains( const box3_t<T>& b) const
    {
        return contains( b.min) && contains( b.max);
    }

    // regular concept
    bool operator==( const box3_t<T>& other) const
    {
        return min == other.min && max == other.max;
    }

    point_type min, max;
};

template<class T>
box3_t<T> transform( const box3_t<T>& box, const matrix44_t<T>& m)
{
    box3_t<T> x( box);
    x.transform( m);
    return x;
}

typedef box3_t<int>     box3i_t;
typedef box3_t<float>   box3f_t;
typedef box3_t<double>  box3d_t;

#ifdef RAMEN_WITH_HALF
    typedef box3_t<half> box3h_t;
#endif

} // math
} // ramen

#endif
