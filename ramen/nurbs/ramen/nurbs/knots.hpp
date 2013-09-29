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

#ifndef RAMEN_NURBS_KNOTS_HPP
#define RAMEN_NURBS_KNOTS_HPP

#include<ramen/nurbs/config.hpp>

#include<cassert>
#include<algorithm>

#include<boost/range/concepts.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>
#include<boost/range/size.hpp>

#include<ramen/nurbs/range_value_type.hpp>

namespace ramen
{
namespace nurbs
{

template<class T, class KnotsRange>
std::size_t find_span( std::size_t degree,
                       std::size_t num_cvs,
                       const KnotsRange& knots,
                       T u)
{
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));

    typedef typename boost::range_iterator<const KnotsRange>::type iterator;
    iterator knots_it( boost::begin( knots));

    assert( boost::size( knots) == num_cvs + degree + 1);

    // special case for closed interval
    if( u == knots_it[num_cvs + 1])
        return num_cvs;

    iterator it = std::upper_bound( knots_it, boost::end( knots), u);
    return std::distance( knots_it, it) - 1;
}

template<class KnotsRange>
bool check_non_decreasing( const KnotsRange& knots)
{
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));

    typedef typename boost::range_iterator<KnotsRange>::type    iterator;
    typedef typename std::iterator_traits<iterator>::value_type knot_type;

    if( boost::size( knots) > 1)
    {
        iterator it( boost::begin( knots));
        knot_type u( *it++);

        for( iterator e( boost::end( knots)); it != e; ++it)
        {
            if( u <= *it)
                return false;

            u = *it;
        }
    }

    return true;
}

} // nurbs
} // ramen

#endif
