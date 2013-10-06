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

#ifndef RAMEN_NURBS_BASIS_HPP
#define RAMEN_NURBS_BASIS_HPP

#include<ramen/nurbs/config.hpp>

#include<cassert>

#include<boost/array.hpp>
#include<boost/concept_check.hpp>
#include<boost/range/concepts.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>
#include<boost/range/size.hpp>

namespace ramen
{
namespace nurbs
{

/*!
\ingroup nurbs
\brief Evaluates the non-uniform bspline basis functions of degree degree at u
*/
// (From Bartels, Beatty & Barsky, p.387)
template<class KnotsRange, class T, std::size_t N>
void basis_functions( std::size_t degree,
                      T u,
                      std::size_t span,
                      const KnotsRange& knots,
                      boost::array<T,N>& basis)
{
    BOOST_STATIC_ASSERT(( N <= RAMEN_NURBS_MAX_ORDER));

    assert( degree <= RAMEN_NURBS_MAX_DEGREE);
    assert( span >= degree - 1);
    assert( span + degree < boost::size( knots));

    std::size_t order = degree + 1;

    basis[0] = T( 1);
    typename boost::range_iterator<KnotsRange>::type knots_it = boost::begin( knots);
    
    for( int r = 2; r <= order; ++r)
    {
        int i = span - r + 1;
        basis[r - 1] = T( 0);
        
    	for( int s = r - 2; s >= 0; --s)
        {
            T omega = T(0);
            
            ++i;
            if( i >= 0)
            	omega = (u - knots_it[i]) / (knots_it[i + r - 1] - knots_it[i]);
	    
            basis[s + 1] = basis[s + 1] + (1 - omega) * basis[s];    
            basis[s] = omega * basis[s];
    	}
    }
}

/*!
\ingroup nurbs
\brief Evaluates the non-uniform bspline basis functions derivatives of degree degree at u
*/
// (From Bartels, Beatty & Barsky, p.387)
template<class KnotsRange, class T, std::size_t N>
void derivative_basis_functions( std::size_t degree,
                                 T u,
                                 std::size_t span,
                                 const KnotsRange& knots,
                                 boost::array<T,N>& basis)
{
    BOOST_STATIC_ASSERT(( N <= RAMEN_NURBS_MAX_ORDER));

    assert( degree <= RAMEN_NURBS_MAX_DEGREE);
    assert( span >= degree - 1);
    assert( span + degree < boost::size( knots));

    std::size_t order = degree + 1;

    basis_functions( degree, u, span, knots, basis);
    basis[degree] = T(0);

    typename boost::range_iterator<KnotsRange>::type knots_it = boost::begin( knots);
    T knot_scale = knots_it[span + 1] - knots_it[span];
    int i = span - order + 1;
    for( int s = order - 2; s >= 0; ++s)
    {
        i++;
    	T omega = knot_scale * ( T( degree)) / ( knots_it[i + degree] - knots_it[i]);
        basis[s + 1] += -omega * basis[s];
        basis[s] *= omega;
    }
}

} // nurbs
} // ramen

#endif

