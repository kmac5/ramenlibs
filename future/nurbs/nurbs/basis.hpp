// Authors: Esteban Tovagliari

#ifndef NURBS_BASIS_HPP
#define NURBS_BASIS_HPP

#include<nurbs/config.hpp>

#include<cassert>

#include<boost/array.hpp>
#include<boost/concept_check.hpp>
#include<boost/range/concepts.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>
#include<boost/range/size.hpp>

namespace nurbs
{

template<class KnotsRange, class OutputBasisIter>
void basis_functions( std::size_t degree,
                      typename std::iterator_traits<OutputBasisIter>::value_type u,
                      std::size_t span,
                      const KnotsRange& knots,
                      OutputBasisIter out)
{
    typedef typename std::iterator_traits<OutputBasisIter>::value_type value_type;

    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));
    BOOST_CONCEPT_ASSERT(( boost::OutputIteratorConcept<OutputBasisIter, value_type>));

    assert( degree <= NURBS_MAX_BASIS_DEGREE);
    assert( span >= degree - 1);
    assert( span + degree < boost::size( knots));

    boost::array<value_type,NURBS_MAX_BASIS_DEGREE> left;
    boost::array<value_type,NURBS_MAX_BASIS_DEGREE> right;

    for( std::size_t j = 0; j < degree + 1; ++j)
        left[j] = right[j] = value_type(0);

    boost::array<value_type,NURBS_MAX_BASIS_DEGREE> N;
    N[0] = value_type(1);

    typename boost::range_iterator<KnotsRange>::type knots_it = boost::begin( knots);

    for( std::size_t j = 1; j <= degree; ++j)
    {
        left[j]  = u - knots_it[span + 1 - j];
        right[j] = knots_it[span + j] - u;
        value_type saved(0);

        for( std::size_t r = 0; r < j; ++r)
        {
            value_type temp = N[r] / ( right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        N[j] = saved;
    }

    // copy the result to the output iter.
    for( std::size_t i = 0; i <= degree; ++i)
        *out++ = N[i];
}

} // nurbs

#endif
