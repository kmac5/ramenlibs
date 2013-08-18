// Authors: Esteban Tovagliari

#ifndef NURBS_EVALUATE_HPP
#define NURBS_EVALUATE_HPP

#include<nurbs/config.hpp>

#include<boost/range/concepts.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>
#include<boost/range/size.hpp>

#include<nurbs/range_value_type.hpp>
#include<nurbs/numeric_ops.hpp>
#include<nurbs/basis.hpp>
#include<nurbs/knots.hpp>

namespace nurbs
{

template<class BasisRange,
         class CVsRange,
         class T>
typename range_value_type<CVsRange>::type
evaluate( std::size_t degree,
          const CVsRange& cvs,
          const BasisRange& N,
          std::size_t span,
          T u)
{
    BOOST_CONCEPT_ASSERT(( boost::ForwardRangeConcept<BasisRange>));
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<CVsRange>));

    assert( boost::size( N) > degree);
    assert( boost::size( cvs) > degree);
    assert( span - degree >= 0);
    assert( span + 1 < boost::size( cvs));

    std::size_t k = span - degree;

    typename boost::range_iterator<CVsRange>::type values_it;
    typename boost::range_iterator<BasisRange>::type N_it;

    typedef typename range_value_type<BasisRange>::type basis_type;
    typedef typename range_value_type<CVsRange>::type value_type;
    value_type result( numeric_ops<value_type, basis_type>::mult_scalar( values_it[k++], *N_it++));

    for( int i = 1, e = degree + 1; i < e; ++i)
        numeric_ops<value_type, basis_type>::add_mult_scalar( result, values_it[k++], *N_it++);

    return result;
}

template<class CVsRange,
         class KnotsRange>
typename range_value_type<CVsRange>::type
evaluate( std::size_t degree,
          const CVsRange& cvs,
          const KnotsRange& knots,
          typename range_value_type<KnotsRange>::type u)
{
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<CVsRange>));
    BOOST_CONCEPT_ASSERT(( boost::RandomAccessRangeConcept<KnotsRange>));

    std::size_t span = find_span( degree, boost::size( cvs), knots, u);

    boost::array<typename range_value_type<KnotsRange>::type, NURBS_MAX_BASIS_DEGREE> N;
    basis_functions( degree, u, span, knots, N.begin());
    return evaluate( degree, cvs, N, span, u);
}

} // nurbs

#endif
