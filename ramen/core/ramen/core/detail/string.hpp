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

#ifndef RAMEN_CORE_DETAIL_STRING_HPP
#define RAMEN_CORE_DETAIL_STRING_HPP

#include<ramen/core/config.hpp>

#include<cassert>

namespace ramen
{
namespace core
{
namespace detail
{

template<class Char>
std::size_t string_length( const Char *str)
{
    assert( str);

    const Char *p = str;
    std::size_t size = 0;

    while( *p)
    {
        ++size;
        ++p;
    }

    return size;
}

template<class Char>
int string_compare( const Char *a, const Char *b)
{
    assert( a);
    assert( b);

    while( true)
    {
        Char ac = *a;
        Char bc = *b;

        if( *a < *b)
            return -1;
        else
            if( *a > *b)
                return 1;

        if( ac == 0 || bc == 0)
            return 0;

        ++a;
        ++b;
    }

    return 0;
}

} // detail
} // core
} // ramen

#endif
