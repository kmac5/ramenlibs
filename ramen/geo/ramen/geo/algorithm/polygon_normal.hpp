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

#ifndef RAMEN_GEO_ALGORITHM_POLYGON_NORMAL_HPP
#define RAMEN_GEO_ALGORITHM_POLYGON_NORMAL_HPP

#include<ramen/geo/config.hpp>

#include<boost/optional.hpp>

#include<ramen/math/point3.hpp>
#include<ramen/math/normal.hpp>

namespace ramen
{
namespace geo
{

template<class PointIter>
boost::optional<math::normal_t<typename std::iterator_traits<PointIter>::value_type::value_type> >
newell_polygon_normal( PointIter start, PointIter end)
{
    typedef typename std::iterator_traits<PointIter>::value_type point_type;
    BOOST_STATIC_ASSERT(( boost::is_same<typename point_type::size_type, boost::mpl::int_<3> >::value));

    typedef typename point_type::value_type value_type;
    typedef math::normal_t<value_type> normal_type;

    std::size_t num_verts = std::distance( start, end);

    if( num_verts < 3)
        return boost::optional<normal_type>();

    math::normal_t<value_type> n( 0, 0, 0);
    for( PointIter it( start); it != end; ++it)
    {
        PointIter next( it + 1);
        if( next == end)
            next = start;

        math::point3_t<value_type> u( *it);
        math::point3_t<value_type> v( *next);
        n.x += ( u.y - v.y) * ( u.z + v.z);
        n.y += ( u.z - v.z) * ( u.x + v.x);
        n.z += ( u.x - v.x) * ( u.y + v.y);
        it = next;
    }

    value_type l = n.length();
    if( l != value_type(0))
    {
        n /= l;
        return n;
    }

    return boost::optional<normal_type>();
}

} // geo
} // ramen

#endif
