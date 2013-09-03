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

#ifndef RAMEN_HASH_HASH_FUNCTION_CONCEPT_HPP
#define RAMEN_HASH_HASH_FUNCTION_CONCEPT_HPP

#include<boost/concept_check.hpp>

namespace ramen
{
namespace hash
{

template <class T>
struct HashFunctionConcept
{
    typedef T                                           hash_function_type;
    typedef typename hash_function_type::digest_type    digest_type;

    BOOST_CONCEPT_USAGE( HashFunctionConcept)
    {
        // check that T has a finalize member.
        digest_type x( hash_fun.finalize());
    }

    hash_function_type hash_fun;
};

} // hash
} // ramen

#endif
