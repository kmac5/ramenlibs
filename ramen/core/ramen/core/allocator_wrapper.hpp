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

#ifndef RAMEN_CORE_ALLOCATOR_WRAPPER_HPP
#define RAMEN_CORE_ALLOCATOR_WRAPPER_HPP

#include<ramen/core/config.hpp>

#include<ramen/core/allocator_interface.hpp>

namespace ramen
{
namespace core
{

template<typename T>
class allocator_wrapper_t
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
        typedef allocator_wrapper_t<Y> other;
    };

    explicit allocator_wrapper_t( allocator_ptr_t alloc) : alloc_( alloc)
    {
        assert( alloc_);
    }

    allocator_wrapper_t( const allocator_wrapper_t& other) : alloc_( other.alloc_)
    {
        std::cout << "Allocator copied" << std::endl;
    }

    ~allocator_wrapper_t()
    {
        std::cout << "Allocator deleted" << std::endl;
    }

    allocator_wrapper_t& operator=( const allocator_wrapper_t& other)
    {
        alloc_ = other.alloc_;
        std::cout << "Allocator assigned" << std::endl;
    }

    const_pointer address( const_reference x) const { return &x;}
    pointer address( reference x) const             { return &x;}

    pointer allocate( size_type n, const void* = 0)
    {
        assert( n <= this->max_size());

        std::cout << "**Allocate called, size = " << n * sizeof( T) << std::endl;
        return reinterpret_cast<T*>( alloc_->allocate( n * sizeof( T)));
    }

    void deallocate( pointer p, size_type)
    {
        std::cout << "**Dellocate called" << std::endl;
        alloc_->deallocate( reinterpret_cast<void*>( p));
    }

    size_type max_size() const _GLIBCXX_USE_NOEXCEPT
    {
        return size_type( -1) / sizeof( T);
    }

    void construct( pointer p, const T& val)
    {
        //std::cout << "****Construct called" << std::endl;
        ::new(( void *) p) T( val);
    }

    void destroy(pointer p)
    {
        //std::cout << "****Destruct called" << std::endl;
        p->~T();
    }

    bool operator==( const allocator_wrapper_t& other) const
    {
        return alloc_ == other.alloc_;
    }

    bool operator!=( const allocator_wrapper_t& other) const
    {
        return !( *this == other);
    }

 private:

    allocator_ptr_t alloc_;
};

} // core
} // ramen

#endif
