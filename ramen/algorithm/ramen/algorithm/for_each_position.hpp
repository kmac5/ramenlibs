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

#ifndef RAMEN_ALGORITHM_FOR_EACH_POSITION_HPP
#define RAMEN_ALGORITHM_FOR_EACH_POSITION_HPP

#include<ramen/algorithm/config.hpp>

#include<boost/bind.hpp>
#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>

namespace ramen
{
namespace algorithm
{

/*!
\ingroup algorithm
\brief Calls f for each iterator in the range [first, last)
*/
template<class InputIterator, class UnaryFunction>
inline void for_each_position( InputIterator first, InputIterator last, UnaryFunction f)
{
    while( first != last)
    {
        f( first);
        ++first;
    }
}

/*!
\ingroup algorithm
\brief Calls f for each iterator in the range [first, last)
*/
template<class InputRange, class UnaryFunction>
inline void for_each_position( InputRange& range, UnaryFunction f)
{
    for_each_position( boost::begin( range), boost::end( range), f);
}

/*!
\ingroup algorithm
\brief Calls f for each iterator in the range [first, last)
*/
template<class InputRange, class UnaryFunction>
inline void for_each_position( const InputRange& range, UnaryFunction f)
{
    for_each_position( boost::begin( range), boost::end( range), f);
}

} // algorithm
} // ramen

#endif
