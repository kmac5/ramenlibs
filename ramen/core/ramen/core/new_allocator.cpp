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

#include<ramen/core/new_allocator.hpp>

#include"tinythread.h"

namespace ramen
{
namespace core
{
namespace
{

tthread::atomic<new_allocator_t*> g_new_allocator_ptr_( 0);
tthread::mutex g_new_alloc_mutex;
allocator_ptr_t g_new_allocator;

} // unnamed

new_allocator_t::new_allocator_t() {}

void *new_allocator_t::allocate( std::size_t size)
{
    return ::operator new( size);
}

void new_allocator_t::deallocate( void *ptr)
{
    ::operator delete( ptr);
}

new_allocator_t *new_allocator_t::create()
{
    return new new_allocator_t();
}

void new_allocator_t::release() const
{
    delete this;
}

allocator_ptr_t global_new_allocator()
{
    new_allocator_t *tmp = g_new_allocator_ptr_.load( tthread::memory_order_consume);
    if( !tmp)
    {
        tthread::lock_guard<tthread::mutex> lock( g_new_alloc_mutex);
        tmp = g_new_allocator_ptr_.load( tthread::memory_order_consume);

        if( !tmp)
        {
            tmp = new_allocator_t::create();
            g_new_allocator_ptr_.store( tmp, tthread::memory_order_release);
            g_new_allocator = allocator_ptr_t( g_new_allocator_ptr_);
        }
    }

    return g_new_allocator;
}

} // core
} // ramen
