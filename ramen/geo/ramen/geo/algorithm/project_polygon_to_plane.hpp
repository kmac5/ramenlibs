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

#ifndef RAMEN_GEO_ALGORITHM_PROJECT_POLYGON_TO_PLANE_HPP
#define RAMEN_GEO_ALGORITHM_PROJECT_POLYGON_TO_PLANE_HPP

#include<ramen/geo/config.hpp>

#include<ramen/math/plane.hpp>

#include<ramen/geo/algorithm/polygon_normal.hpp>

namespace ramen
{
namespace geo
{

template<class T, class PointIter, class OutPointIter>
void project_polygon_to_plane( const math::plane_t<T>& plane,
                               PointIter start,
                               PointIter end,
                               OutPointIter result)
{
    typedef typename std::iterator_traits<PointIter>::value_type    point3_type;

    BOOST_STATIC_ASSERT(( boost::is_same<typename point3_type::size_type, boost::mpl::int_<3> >::value));
    BOOST_STATIC_ASSERT(( boost::is_same<T, typename point3_type::value_type>::value));

    for( PointIter it( start); it != end; ++it)
        *result++ = plane.nearest_point_on_plane( *it);
}

template<class PointIter, class OutPointIter>
bool make_polygon_planar( PointIter start, PointIter end, OutPointIter result)
{
    typedef typename std::iterator_traits<PointIter>::value_type    point3_type;
    typedef typename point3_type::normal_type                       normal_type;
    typedef typename point3_type::value_type                        scalar_type;

    boost::optional<normal_type> n( newell_polygon_normal( start, end));
    
    if( n)
    {
        math::plane_t<scalar_type> plane( n, *start);
        project_polygon_to_plane( plane, start, end, result);
        return true;
    }

    return false;
}

} // geo
} // ramen

#endif
