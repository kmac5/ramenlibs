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

#ifndef RAMEN_ARRAYS_ARRAY_HPP
#define RAMEN_ARRAYS_ARRAY_HPP

#include<ramen/arrays/array_fwd.hpp>

#include<cstddef>
#include<cassert>

#include<boost/move/move.hpp>

#include<ramen/core/final.hpp>
#include<ramen/core/types.hpp>
#include<ramen/core/allocator_interface.hpp>

#include<ramen/arrays/detail/array_interface_fwd.hpp>

namespace ramen
{
namespace arrays
{

/*!
\brief A polymorphic array of elements.
*/
class RAMEN_ARRAYS_API array_t : RAMEN_CORE_FINAL( array_t)
{
    BOOST_COPYABLE_AND_MOVABLE( array_t)

public:

    typedef std::size_t size_type;

    array_t( core::type_t type = core::float_k, size_type n = 0);
    ~array_t();

    // Copy constructor
    array_t( const array_t& other);

    // Copy assignment
    array_t& operator=( BOOST_COPY_ASSIGN_REF( array_t) other)
    {
        array_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    array_t( BOOST_RV_REF( array_t) other) : model_( 0)
    {
        assert( other.model_);

        swap( other);
    }

    // Move assignment
    array_t& operator=( BOOST_RV_REF( array_t) other)
    {
        assert( other.model_);

        swap( other);
        return *this;
    }

    void swap( array_t& other);

    core::type_t type() const;

    bool empty() const;
    size_type size() const;
    size_type capacity() const;

    void clear();
    void reserve( size_type n);
    void erase( size_type start, size_type end);
    void erase( size_type pos);
    void shrink_to_fit();

    void copy_element( size_type from, size_type to);

    // for RegularConcept
    bool operator==( const array_t& other) const;
    bool operator!=( const array_t& other) const;

private:

    template<class>
    friend struct array_ref_traits;

    template<class>
    friend class const_array_ref_t;

    template<class>
    friend class array_ref_t;

    void init( core::type_t type, size_type n, const core::allocator_ptr_t& alloc);

    template<class T>
    void do_init( const core::allocator_ptr_t& alloc);

    void do_init_with_string( const core::allocator_ptr_t& alloc);

    // used by array_refs
    const detail::array_interface_t *model() const;
    detail::array_interface_t *model();

    void push_back( const void *x);
    void resize( size_type n);

    detail::array_interface_t *model_;
};

inline void swap( array_t& x, array_t& y)
{
    x.swap( y);
}

/// Sets the default array allocator. Warning: not thread safe.
RAMEN_ARRAYS_API void set_default_array_allocator( const core::allocator_ptr_t& alloc);

} // arrays
} // ramen

#endif
