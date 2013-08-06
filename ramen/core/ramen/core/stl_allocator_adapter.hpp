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

#ifndef RAMEN_CORE_STL_ALLOCATOR_ADAPTER_HPP
#define RAMEN_CORE_STL_ALLOCATOR_ADAPTER_HPP

#include<ramen/core/config.hpp>

#include<ramen/core/allocator_interface.hpp>

namespace ramen
{
namespace core
{

template<typename T>
class stl_allocator_adapter_t
{
public:

    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;
    typedef T*              pointer;
    typedef const T*        const_pointer;
    typedef T&              reference;
    typedef const T&        const_reference;
    typedef T               value_type;

    template<typename Y>
    struct rebind
    {
        typedef stl_allocator_adapter_t<Y> other;
    };

    explicit stl_allocator_adapter_t( allocator_ptr_t alloc) : alloc_( alloc)
    {
        assert( alloc_);
    }

    const_pointer address( const_reference x) const { return &x;}
    pointer address( reference x) const             { return &x;}

    pointer allocate( size_type n, const void* = 0)
    {
        assert( n <= this->max_size());

        return reinterpret_cast<T*>( alloc_->allocate( n * sizeof( T)));
    }

    void deallocate( pointer p, size_type)
    {
        alloc_->deallocate( reinterpret_cast<void*>( p));
    }

    size_type max_size() const
    {
        return size_type( -1) / sizeof( T);
    }

    void construct( pointer p, const T& val)
    {
        ::new(( void *) p) T( val);
    }

    void destroy(pointer p)
    {
        p->~T();
    }

    bool operator==( const stl_allocator_adapter_t& other) const
    {
        return alloc_ == other.alloc_;
    }

    bool operator!=( const stl_allocator_adapter_t& other) const
    {
        return !( *this == other);
    }

 private:

    allocator_ptr_t alloc_;
};

} // core
} // ramen

#endif
