// Authors: Esteban Tovagliari

#ifndef NURBS_APPROX_EQUAL_HPP
#define NURBS_APPROX_EQUAL_HPP

#include<nurbs/config.hpp>

namespace nurbs
{

template<class T>
bool approx_equal( T x, T y)
{
    return x == y;
}

template<class T>
bool approx_less( T x, T y)
{
    return x < y;
}

template<class T>
bool approx_less_equal( T x, T y)
{
    return x <= y;
}

template<class T>
bool approx_greater( T x, T y)
{
    return x > y;
}

template<class T>
bool approx_greater_equal( T x, T y)
{
    return x >= y;
}

} // nurbs

#endif
