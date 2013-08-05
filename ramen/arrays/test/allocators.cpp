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

#include<gmock/gmock.h>
#include<gtest/gtest.h>

//#include<ramen/arrays/apply_visitor.hpp>

#include<boost/shared_ptr.hpp>
#include<boost/container/vector.hpp>

#include<iostream>

namespace ramen
{
namespace core
{
} // core

namespace arrays
{

class allocator_interface_t
{
public:

    virtual ~allocator_interface_t() {}

    virtual void *allocate( std::size_t size) = 0;
    virtual void deallocate( void *ptr) = 0;

protected:

    allocator_interface_t() {}

private:

    // non-copyable
    allocator_interface_t( const allocator_interface_t&);
    allocator_interface_t& operator=( const allocator_interface_t&);
};

class new_allocator_t : public allocator_interface_t
{
public:

    virtual void *allocate( std::size_t size)
    {
        return ::operator new( size);
    }

    virtual void deallocate( void *ptr)
    {
        ::operator delete( ptr);
    }
};

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

    explicit allocator_wrapper_t( boost::shared_ptr<allocator_interface_t> alloc) : alloc_( alloc)
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

    boost::shared_ptr<allocator_interface_t> alloc_;
 };


} // arrays
} // ramen

using namespace ramen::core;
using namespace ramen::arrays;

TEST( ArrayAllocators, All)
{
    boost::shared_ptr<allocator_interface_t> alloc( new new_allocator_t());
    allocator_wrapper_t<int> alloc_wrapper( alloc);

    typedef boost::container::vector<int, allocator_wrapper_t<int> > vector_type;

    vector_type vec( alloc_wrapper);
    boost::container::vector<int> vec_no_alloc;

    for( int i = 0; i < 100; ++i)
    {
        vec.push_back( i);
        vec_no_alloc.push_back( i);
    }

    EXPECT_EQ( vec.size(), vec_no_alloc.size());
    EXPECT_EQ( vec.capacity(), vec_no_alloc.capacity());

    vector_type vec2 = vec;
    vec.swap( vec2);

    vec.clear();
    vec.reserve( 100);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleMock( &argc, argv);
    return RUN_ALL_TESTS();
}
