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

#ifndef RAMEN_MATH_BOX2_HPP
#define RAMEN_MATH_BOX2_HPP

#include<ramen/math/config.hpp>

#include<boost/operators.hpp>

#include<ramen/math/point2.hpp>

namespace ramen
{
namespace math
{

/*!
\brief A two dimensional bounding box.
*/
template<class T>
class box2_t : boost::equality_comparable<box2_t<T> >
{
public:

    typedef T               value_type;
    typedef point2_t<T>     point_type;
    typedef vector2_t<T>    vector_type;

    box2_t()
    {
        reset();
    }

    box2_t( const point_type& p) : min( p), max( p) {}

    box2_t( const point_type& pmin, const point_type& pmax) : min( pmin), max( pmax) {}

    void reset()
    {
        min.x = min.y = std::numeric_limits<T>::max();
        max.x = max.y = -std::numeric_limits<T>::max();
    }

    bool empty() const
    {
        return max.x < min.x || max.y < min.y;
    }

    vector_type size() const
    {
        return max - min;
    }

    point_type center() const
    {
        return min + ( size() * T(0.5));
    }

    void offset_by( const vector2_t<T>& v)
    {
        min += v;
        max += v;
    }

    bool operator==( const box2_t<T>& other) const
    {
        return min == other.min && max == other.max;
    }

    point_type min, max;
};

typedef box2_t<int>     box2i_t;
typedef box2_t<float>   box2f_t;
typedef box2_t<double>  box2d_t;

#ifdef RAMEN_WITH_HALF
    typedef box2_t<half> box2h_t;
#endif

} // math
} // ramen

#endif
