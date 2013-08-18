
// Authors: Esteban Tovagliari

#ifndef NURBS_KNOTS_HPP
#define NURBS_KNOTS_HPP

#include<nurbs/config.hpp>

#include<cassert>

#include<boost/range/concepts.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>

#include<nurbs/range_value_type.hpp>
#include<nurbs/approx_equal.hpp>

namespace nurbs
{

template<class PointsRange, class DistanceFun, class OutKnotsIter>
void chord_length_parameterization( const PointsRange& points,
                                    DistanceFun dist_fun,
                                    OutKnotsIter knots)
{
    typedef typename std::iterator_traits<OutKnotsIter>::value_type knot_type;

    BOOST_CONCEPT_ASSERT(( boost::ForwardRangeConcept<PointsRange>));
    BOOST_CONCEPT_ASSERT(( boost::OutputIteratorConcept<OutKnotsIter, knot_type>));

    typename boost::range_iterator<PointsRange>::type point_it( boost::begin( points));
    std::size_t num_points( boost::size( points));

    knot_type sum_lengths( 0);

    for( std::size_t i = 1; i != num_points; ++i)
        sum_lengths += dist_fun( points_it[i], points_it[i-1]);

    knot_type prev_knot = knot_type( 0);
    knot_type this_knot = knot_type( 0);
    *knots++ = prev_knot;

    if( sum_lengths > 0)
    {
        for( std::size_t i = 1; i < num_points - 1; ++i)
        {
            this_knot = prev_knot + dist_fun( point_it[i], point_it[i-1]) / sum_lengths;
            prev_knot = this_knot;
            *knots++ = this_knot;
        }
    }
    else
    {
        for( std::size_t i = 1; i < num_points - 1; ++i)
            *knots++ = knot_type( i) / knot_type( num_points - 1);
    }

    *knots = knot_type( 1.0);
}

template<class PointsRange, class DistanceFun, class OutKnotsIter>
void chord_length_periodic_parameterization( std::size_t degree,
                                             const PointsRange& points,
                                             DistanceFun dist_fun,
                                             OutKnotsIter knots)
{
    typedef typename std::iterator_traits<OutKnotsIter>::value_type knot_type;

    BOOST_CONCEPT_ASSERT(( boost::ForwardRangeConcept<PointsRange>));
    BOOST_CONCEPT_ASSERT(( boost::OutputIteratorConcept<OutKnotsIter, knot_type>));

/*
int i ;
T d = 0.0 ;

ub.resize(Qw.n()) ;
ub[0] = 0 ;
for(i=1;i<=ub.n()-deg;i++){
  d += norm(Qw[i]-Qw[i-1]) ;
}
if(d>0){
  for(i=1;i<ub.n();++i)
    ub[i] = ub[i-1] + norm(Qw[i]-Qw[i-1]);
  // Normalization
  for(i=0;i<ub.n();++i)
    ub[i]/=  d;
}
else
  for(i=1;i<ub.n();++i)
    ub[i] = (T)(i)/(T)(ub.n()-2) ;

return d ;
*/
}

} // nurbs

#endif
