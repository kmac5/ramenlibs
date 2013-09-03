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

#ifndef RAMEN_HASH_HASH_GENERATOR_HPP
#define RAMEN_HASH_HASH_GENERATOR_HPP

#include<ramen/hash/Concepts/HashFunctionConcept.hpp>

#include<ramen/hash/hash_traits.hpp>

namespace ramen
{
namespace hash
{

template<class HashFun>
class generator_t
{
    BOOST_CONCEPT_ASSERT(( HashFunctionConcept<HashFun>));

public:

    typedef HashFun                                     hash_function_type;
    typedef typename hash_function_type::digest_type    digest_type;

    generator_t()
    {
    }

    template<class T>
    void operator()( const T& x)
    {
        hash_traits<T, HashFun>::hash( x, hash_fun_);
    }

    digest_type finalize()
    {
        return hash_fun_.finalize();
    }

private:

    // non-copyable
    generator_t( const generator_t&);
    generator_t& operator=( const generator_t&);

    HashFun hash_fun_;
};

} // hash
} // ramen

#endif
