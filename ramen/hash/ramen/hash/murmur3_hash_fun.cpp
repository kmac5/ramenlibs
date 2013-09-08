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

#include<ramen/hash/murmur3_hash_fun.hpp>

#include<vector>

#include<ramen/hash/extern/MurmurHash3.h>

namespace ramen
{
namespace hash
{

struct murmur3_hash_fun::impl
{
    explicit impl( boost::uint32_t seed) : seed_( seed)
    {
    }

    void append( const void *data, std::size_t size)
    {
        storage_.reserve( storage_.size() + size);
        std::copy( reinterpret_cast<const boost::uint8_t*>( data),
                   reinterpret_cast<const boost::uint8_t*>( data) + size,
                   std::back_inserter( storage_));
    }

    murmur3_hash_fun::digest_type finalize()
    {
        murmur3_hash_fun::digest_type result;

        const void *data = 0;
        if( !storage_.empty())
            data = reinterpret_cast<const void*>( &storage_[0]);

        MurmurHash3_x64_128( data,
                             storage_.size(),
                             seed_,
                             reinterpret_cast<void*>( result.begin()));
        return result;
    }

    boost::uint32_t seed_;
    std::vector<boost::uint8_t> storage_;
};

murmur3_hash_fun::murmur3_hash_fun( boost::uint32_t seed)
{
    pimpl_ = new impl( seed);
}

murmur3_hash_fun::~murmur3_hash_fun()
{
    delete pimpl_;
}

void murmur3_hash_fun::swap( murmur3_hash_fun& other)
{
    std::swap( pimpl_, other.pimpl_);
}

void murmur3_hash_fun::operator()( const void *data, std::size_t size)
{
    pimpl_->append( data, size);
}

murmur3_hash_fun::digest_type murmur3_hash_fun::finalize()
{
    return pimpl_->finalize();
}

} // hash
} // ramen
