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

#include<ramen/cuda/host_allocator.hpp>

#include<boost/integer.hpp>

#include<ramen/cuda/cudart.hpp>

namespace ramen
{
namespace cuda
{

host_allocator_t::host_allocator_t( unsigned int flags) : core::allocator_interface_t()
{
    flags_ = flags;
}

void *host_allocator_t::allocate( std::size_t size)
{
    return reinterpret_cast<void*>( cuda_host_alloc<boost::uint8_t>( size, flags_));
}

void host_allocator_t::deallocate( void *ptr)
{
    return cuda_free_host( ptr);
}

host_allocator_t *host_allocator_t::create( unsigned int flags)
{
    return new host_allocator_t( flags);
}

void host_allocator_t::release() const
{
    delete this;
}

} // cuda
} // ramen
