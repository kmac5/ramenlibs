
// Authors: Esteban Tovagliari

#ifndef NURBS_RANGE_VALUE_TYPE_HPP
#define NURBS_RANGE_VALUE_TYPE_HPP

#include<nurbs/config.hpp>

#include<boost/range/iterator.hpp>

namespace nurbs
{

template<class Range>
class range_value_type
{
    typedef typename boost::range_iterator<Range>::type iterator;

public:

    typedef typename std::iterator_traits<iterator>::value_type type;
};

} // nurbs

#endif
