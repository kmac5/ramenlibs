
// Authors: Esteban Tovagliari

#ifndef NURBS_KNOTS_HPP
#define NURBS_KNOTS_HPP

#include<nurbs/config.hpp>

#include<cassert>

#include<boost/range/concepts.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>
#include<boost/range/size.hpp>

#include<nurbs/range_value_type.hpp>
#include<nurbs/approx_equal.hpp>

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

    assert( boost::size( knots) == num_cvs + degree + 1);

    iterator knots_it( boost::begin( knots));

    if( approx_greater_equal( u, knots_it[num_cvs]))
        return num_cvs - 1;

    if( approx_less_equal( u, knots_it[degree]))
        return degree;

    // binary search
    std::size_t low = 0;
    std::size_t high = num_cvs + 1;
    std::size_t mid = ( low + high) / 2;

    while( approx_less( u, knots_it[mid]) || approx_greater_equal( u, knots_it[mid + 1]))
    {
        if( approx_less( u, knots_it[mid]))
            high = mid;
        else
            low = mid;

        mid = ( low + high) / 2;
    }

    return mid;
}

template<class T, class KnotsRange>
std::size_t find_knot( std::size_t degree,
                       std::size_t num_cvs,
                       const KnotsRange& knots,
                       T u)
{
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));

    assert( boost::size( knots) == num_cvs + degree + 1);

    typedef typename boost::range_iterator<KnotsRange>::type iterator;
    iterator knots_it( boost::begin( knots));

    if( approx_equal( u, knots_it[num_cvs]))
        return num_cvs;

    for( std::size_t i = degree + 1; i < num_cvs; ++i)
    {
        if( approx_greater( knots_it[i], u))
            return i-1;
    }

    return 0;
}

template<class KnotsRange>
std::size_t knot_multiplicity( std::size_t degree,
                               std::size_t knot_index,
                               const KnotsRange& knots)
{
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));

    assert( boost::size( knots) > degree + 1);
    assert( knot_index > degree);

    typedef typename boost::range_iterator<KnotsRange>::type iterator;
    iterator knots_it( boost::begin( knots));

    std::size_t multiplicity = 1;

    for( std::size_t i = knot_index; i > degree + 1; --i)
    {
        if( approx_less_equal( knots_it[i], knots_it[i-1]))
            ++multiplicity;
        else
            break;
    }

    return multiplicity;
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
            if( approx_less( u, *it))
                return false;

            u = *it;
        }
    }

    return true;
}

// TODO: this needs a better name.
/*
template<class CVsRange, class KnotsRange, class OutCVsIter>
void insert_knot( size_t degree,
                  const CVsRange& cvs,
                  const KnotsRange& knots,
                  typename range_value_type<KnotsRange>::type new_knot,
                  std::size_t multiplicity,
                  OutCVsIter out_cvs)
{
    typedef typename std::iterator_traits<OutCVsIter>::value_type value_type;

    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));
    BOOST_CONCEPT_ASSERT(( boost::OutputIteratorConcept<OutCVsIter, value_type>));

    // TODO: implement this...
    typedef typename boost::range_iterator<const CVsRange>::type cvs_iterator;
    for( cvs_iterator it( boost::begin( cvs)), e( boost::end( cvs)); it != e; ++it)
        *out_cvs++ = *it;
}
*/

} // nurbs

#endif
