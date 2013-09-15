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

#ifndef RAMEN_MATH_LINE_HPP
#define RAMEN_MATH_LINE_HPP

#include<ramen/math/config.hpp>

#include<boost/operators.hpp>

namespace ramen
{
namespace math
{

/*!
\brief Infinite line.
*/
template<class Point>
class line_t : boost::equality_comparable<line_t<Point> >
{
public:

    typedef Point                               point_type;
    typedef typename point_type::value_type     scalar_type;
    typedef typename point_type::vector_type    vector_type;

    line_t() {}

    line_t( const point_type& origin, const vector_type& dir) : o( origin), d( dir)
    {
    }

    line_t( const point_type& p, const point_type& q) : o( p), d( q - p)
    {
    }

    void normalize_direction()
    {
        d = d.normalize();
    }

    point_type operator()( scalar_type t) const
    {
        return o + t * d;
    }

    bool operator==( const line_t<Point>& other)
    {
        return o == other.o && d == other.d;
    }

    point_type o;
    vector_type d;
};

typedef line_t<float>     linef_t;
typedef line_t<double>    lined_t;

} // math
} // ramen

#endif
