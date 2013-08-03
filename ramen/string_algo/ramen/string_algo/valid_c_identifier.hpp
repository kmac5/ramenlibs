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

#ifndef RAMEN_STRING_ALGO_VALID_C_IDENTIFIER_HPP
#define RAMEN_STRING_ALGO_VALID_C_IDENTIFIER_HPP

#include<ramen/string_algo/config.hpp>

#include<cctype>
#include<string.h>

#include<boost/range/begin.hpp>
#include<boost/range/end.hpp>

namespace ramen
{
namespace string_algo
{
namespace detail
{

inline bool check_char( int c, bool numbers_valid)
{
    if( isalpha( c))
        return true;

    if( numbers_valid && isdigit( c))
        return true;

    if( c == '_')
        return true;

    return false;
}

} // detail

template<class Iterator>
bool is_valid_c_identifier( Iterator first, Iterator last)
{
    if( first == last)
        return false;

    Iterator it( first);
    if( !detail::check_char( *it, false))
        return false;

    for( ; it != last; ++it)
    {
        if( !detail::check_char( *it, true))
            return false;
    }

    return true;
}

template<class Range>
bool is_valid_c_identifier( const Range& str)
{
    return is_valid_c_identifier( boost::begin( str), boost::end( str));
}

inline bool is_valid_c_identifier( const char *str)
{
    return is_valid_c_identifier( str, str + strlen( str) + 1);
}

} // string_algo
} // ramen

#endif
