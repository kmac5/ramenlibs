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

#ifndef RAMEN_CONTAINERS_PTR_VECTOR_HPP
#define RAMEN_CONTAINERS_PTR_VECTOR_HPP

#include<ramen/containers/config.hpp>

#include<cassert>

#include<boost/ptr_container/ptr_vector.hpp>

#include<ramen/core/memory.hpp>

namespace ramen
{
namespace containers
{

/*!
  \brief Container of pointers. The vector manages the lifetime of the pointers inserted into it.
 */
template<class T,
         class C = boost::heap_clone_allocator,
         class A = std::allocator<void*> >
class ptr_vector_t : public boost::ptr_vector<T, C, A>
{
    typedef boost::ptr_vector<T, C, A> base_container_type;

public:

    typedef typename base_container_type::iterator iterator;
    typedef typename base_container_type::const_iterator const_iterator;

    /// Constructor
    ptr_vector_t() : base_container_type() {}

    /// Adds a pointer to the vector.
    /// The vector takes ownership of the pointer.
    void push_back( core::auto_ptr_t<T> x)
    {
        assert( x.get());

        push_back( x.release());
    }

    /// Adds a pointer to the vector.
    /// The vector takes ownership of the pointer.
    void push_back( T *x)
    {
        assert( x);

        base_container_type::push_back( x);
    }

    /// Returns @c true if the vector contains the given pointer.
    bool contains_ptr( const T *x) const
    {
        return find_pointer( x) != this->end();
    }

    /// Removes the pointer from the container and transfer ownership to the caller.
    /// Returns an empty unique ptr is the vector didn't contain the pointer.
    core::auto_ptr_t<T> release_ptr( T *x)
    {
        for( iterator it( this->begin()), e( this->end()); it != e; ++it)
        {
            if( &( *it) == x)
            {
                typename base_container_type::auto_type ptr = this->release( it);
                T *elem = ptr.release();
                return core::auto_ptr_t<T>( elem);
            }
        }

        return core::auto_ptr_t<T>();
    }

    /// Transfers ownership of the pointer to another vector.
    void transfer_ptr( T *x, ptr_vector_t<T, C, A>& other)
    {
        core::auto_ptr_t<T> p( release_ptr( x));
        other.push_back( p);
    }

private:

    // disable this
    void push_back( std::auto_ptr<T> x);

    const_iterator find_pointer( const T *x) const
    {
        for( const_iterator it( this->begin()), e( this->end()); it != e; ++it)
        {
            if( &(*it) == x)
                return it;
        }
        return this->end();
    }

    iterator find_pointer( T *x)
    {
        for( iterator it( this->begin()), e( this->end()); it != e; ++it)
        {
            if( &(*it) == x)
                return it;
        }
        return this->end();
    }
};

} // containers
} // ramen

#endif
