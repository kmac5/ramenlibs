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

#ifndef RAMEN_BEZIER_BERNSTEIN_HPP
#define	RAMEN_BEZIER_BERNSTEIN_HPP

#include<ramen/bezier/config.hpp>

#include<boost/array.hpp>

namespace ramen
{
namespace bezier
{

/*!
\ingroup bezier
\brief Evaluates the Bernstein polynomial i of degree Degree at u
*/
template<class T, int Degree>
T bernstein( int i, T u)
{
    T tmp[Degree + 1];

    for( int j = 0; j <= Degree; ++j)
		tmp[j] = T(0);

    tmp[Degree - i] = T(1);
    T u1 = T(1) - u;

    for( int k = 1; k <= Degree; ++k)
    {
		for( int j = Degree; j >= k; --j)
		    tmp[j] = u1 * tmp[j] + u * tmp[j-1];
    }

    return tmp[Degree];
}

/*!
\ingroup bezier
\brief Evaluates all Bernstein polynomials of degree Degree at u
*/
template<class T, int Degree>
void all_bernstein( T u, boost::array<T, Degree+1>& B)
{
    B[0] = T(1);
    T u1 = T(1) - u;

    for( int j = 1; j <= Degree; ++j)
    {
        T saved = T(0);

        for( int k = 0; k < j; ++k)
        {
            T temp = B[k];
            B[k] = saved + u1 * temp;
            saved = u * temp;
        }

        B[j] = saved;
    }
}

} // bezier
} // ramen

#endif
