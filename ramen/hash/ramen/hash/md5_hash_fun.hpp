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

#ifndef RAMEN_HASH_MD5_HASH_FUN_HPP
#define RAMEN_HASH_MD5_HASH_FUN_HPP

#include<ramen/hash/config.hpp>

#include<boost/move/move.hpp>

namespace ramen
{
namespace hash
{

class RAMEN_HASH_API md5_hash_fun
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( md5_hash_fun)

public:

    // totally made up...
    typedef int digest_type;

    md5_hash_fun();
    ~md5_hash_fun();

    // Move constructor
    md5_hash_fun( BOOST_RV_REF( md5_hash_fun) other) : pimpl_( 0)
    {
        swap( other);
    }

    // Move assignment
    md5_hash_fun& operator=( BOOST_RV_REF( md5_hash_fun) other)
    {
        swap( other);
        return *this;
    }

    void swap( md5_hash_fun& other);

    void operator()( const void *data, std::size_t size)
    {
        // TODO: implement...
    }

    // also made up
    digest_type finalize()
    {
        return 0;
    }

private:

    struct impl;
    impl *pimpl_;
};

} // hash
} // ramen

#endif
