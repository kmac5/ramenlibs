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

#ifndef RAMEN_STRING_ALGO_INCREMENT_END_DIGITS_HPP
#define RAMEN_STRING_ALGO_INCREMENT_END_DIGITS_HPP

#include<ramen/string_algo/config.hpp>

#include<boost/lexical_cast.hpp>

namespace ramen
{
namespace string_algo
{
namespace detail
{

template<class String>
String get_end_digits( const String& str)
{
    try
    {
        // this is a quick fix, if all str are nums
        // TODO: not sure, this is needed.
        int num = boost::lexical_cast<int>( str);
        return str;
    }
    catch( ...) {}

    std::size_t n = str.length() - 1;

    while( n != 0)
    {
        char c = str[n];

        if( !isdigit( c))
            break;

        --n;
    }

    return String( str, n + 1, str.length());
}

} // detail

template<class String>
String increment_end_digits( const String& s)
{
    String str = s;
    String num_str = detail::get_end_digits( str);

    if( num_str.empty())
        str.append( "_2");
    else
    {
        String base( str, 0, str.length() - num_str.length());
        int num = boost::lexical_cast<int>( num_str);
        num_str = boost::lexical_cast<String>( ++num);
        str = base;
        str.append( num_str);
    }

    return str;
}

} // string_algo
} // ramen

#endif
