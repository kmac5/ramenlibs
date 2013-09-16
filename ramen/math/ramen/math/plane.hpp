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

#ifndef RAMEN_MATH_PLANE_HPP
#define RAMEN_MATH_PLANE_HPP

#include<ramen/math/config.hpp>

#include<boost/optional.hpp>

#include<ramen/math/normal.hpp>
#include<ramen/math/point3.hpp>
#include<ramen/math/line.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Infinite plane.
*/
template<class T>
class plane_t : boost::equality_comparable<plane_t<T> >
{
public:

    typedef T               value_type;
    typedef normal_t<T>     normal_type;
    typedef vector3_t<T>    vector_type;
    typedef point3_t<T>     point_type;

    plane_t() {}

    plane_t( const normal_t<T>& normal, T distance) : n( normal), d( distance)
    {
    }

    plane_t( const normal_t<T>& normal, const point3_t<T>& p) : n( normal), d( T(0))
    {
        d = -(*this)( p);
    }

    plane_t( const point3_t<T>& p, const point3_t<T>& q, const point3_t<T>& r) : d( T(0))
    {
        vector3_t<T> pq = q - p;
        vector3_t<T> pr = r - p;
        n = cross( pq, pr).normalize();
        d = -(*this)( p);
    }

    T operator()( const point3_t<T>& p) const
    {
        return ( p.x * n.x) + ( p.y * n.y) + ( p.z * n.z) + d;
    }

    bool operator==( const plane_t<T>& other)
    {
        return n == other.n && d == other.d;
    }

    boost::optional<value_type> intersection_param( const line_t<point3_t<T> >& line)
    {
        T denom = dot( n, line.d);
        if( denom == T(0))
            return boost::optional<value_type>();

        T num = -d - ( n.x * line.o.x) - ( n.y * line.o.y) - ( n.z * line.o.z);
        return num / denom;
    }

    boost::optional<point3_t<T> > intersection_point( const line_t<point3_t<T> >& line)
    {
        boost::optional<value_type> t( intersection_param( line));

        if( t)
            return (*this)( t.get());

        return boost::optional<point_type>();
    }

    boost::optional<point3_t<T> > nearest_point_on_plane( const point3_t<T>& p)
    {
        line_t<point3_t<T> > line( p, n);
        return intersection_point( p);
    }

    normal_t<T> n;
    T d;
};

typedef plane_t<float>     planef_t;
typedef plane_t<double>    planed_t;

} // math
} // ramen

#endif
