
// Authors: Esteban Tovagliari

#ifndef NURBS_NUMERIC_OPS_HPP
#define NURBS_NUMERIC_OPS_HPP

#include<nurbs/config.hpp>

namespace nurbs
{

template<class T, class S>
struct numeric_ops
{
    static T mult_scalar( T x, S s)
    {
        return x * s;
    }

    static void add_mult_scalar( T& x, T y, S s)
    {
        x += mult_scalar( y, s);
    }
};

} // nurbs

#endif
