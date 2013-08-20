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

#ifndef RAMEN_CORE_MEMORY_HPP
#define RAMEN_CORE_MEMORY_HPP

#include<ramen/core/config.hpp>

#include<stdint.h>

#include<boost/move/move.hpp>
#include<boost/static_assert.hpp>

#include<cassert>

#include<boost/move/move.hpp>
#include<boost/swap.hpp>

namespace ramen
{
namespace core
{

namespace detail
{

struct new_delete_t
{
    void *(*new_)( std::size_t);
    void  (*delete_)( void*);
};

extern const new_delete_t g_local_new_delete;

} // detail

template<class T>
struct pointer_traits;

template<class T>
struct pointer_traits<T*>
{
    typedef T                   element_type;
    typedef T*                  pointer_type;
    typedef const pointer_type  const_pointer_type;

    static bool empty_ptr( T *x)  { return x == 0;}
    static void delete_ptr( T *x) { delete x;}
};

template<class T>
struct pointer_traits<T(*)[]>
{
    typedef T                   element_type;
    typedef T*                  pointer_type;
    typedef const pointer_type  const_pointer_type;

    static bool empty_ptr( T *x)  { return x == 0;}
    static void delete_ptr( T *x) { delete[] x;}
};

/*!
    \ingroup core
    \brief auto_resource_t offers similar functionality to std::auto_ptr for resources for
    which the pointer type is <i>opaque</i> refered to by a non-pointer type.
*/
template<class T>
class auto_resource_t
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( auto_resource_t)

    // for safe bool
    operator int() const;

public:

    typedef pointer_traits<T>                         traits_type;
    typedef typename traits_type::element_type        element_type;
    typedef typename traits_type::pointer_type        pointer_type;
    typedef typename traits_type::const_pointer_type  const_pointer_type;

    explicit auto_resource_t( pointer_type ptr = 0) throw() : ptr_( 0)
    {
        reset( ptr);
    }

    ~auto_resource_t()
    {
        delete_contents();
    }

    auto_resource_t( BOOST_RV_REF( auto_resource_t) other) throw()  : ptr_( other.release()) {}

    auto_resource_t& operator=( BOOST_RV_REF( auto_resource_t) other)
    {
        swap( other);
        return *this;
    }

    pointer_type get() const throw() { return ptr_;}

    pointer_type release() throw()
    {
        pointer_type rv = ptr_;
        ptr_ = 0;
        return rv;
    }

    void reset( pointer_type ptr = 0) throw()
    {
        if( ptr_ != ptr)
        {
            delete_contents();
            ptr_ = ptr;
        }
    }

    void swap( auto_resource_t& other)
    {
        boost::swap( ptr_, other.ptr_);
    }

    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return !traits_type::empty_ptr( ptr_);}
    bool operator!() const throw();

private:

    void delete_contents()
    {
        if( ptr_)
        {
            traits_type::delete_ptr( ptr_);
            ptr_ = 0;
        }
    }

    pointer_type ptr_;
};

template<class T>
inline void swap( auto_resource_t<T>& x, auto_resource_t<T>& y)
{
    x.swap( y);
}

/*!
    \ingroup core
    \brief An improved std::auto_ptr like smart pointer.
*/
template<class T>
class auto_ptr_t : public auto_resource_t<T*>
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( auto_ptr_t)

    // for safe bool
    operator int() const;

public:

    typedef pointer_traits<T*>                        traits_type;
    typedef typename traits_type::element_type        element_type;
    typedef typename traits_type::pointer_type        pointer_type;
    typedef typename traits_type::const_pointer_type  const_pointer_type;

    explicit auto_ptr_t( pointer_type ptr = 0) throw() : auto_resource_t<T*>( ptr) {}

    auto_ptr_t( BOOST_RV_REF( auto_ptr_t) other) throw() : auto_resource_t<T*>( boost::move( static_cast<auto_resource_t<T*>&>( other)))
    {
    }

    auto_ptr_t& operator=( BOOST_RV_REF( auto_ptr_t) other)
    {
        auto_resource_t<T*>::operator=( boost::move(static_cast<auto_resource_t<T*>&>( other)));
        return *this;
    }

    element_type& operator*() const throw()
    {
        assert( this->get());

        return *this->get();
    }

    pointer_type operator->() const throw()
    {
        assert( this->get());

        return this->get();
    }

    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return !traits_type::empty_ptr( this->get());}
    bool operator!() const throw();
};

template<class T>
class aligned_storage_t
{
    BOOST_COPYABLE_AND_MOVABLE( aligned_storage_t)

public:

    aligned_storage_t()
    {
        init();
    }

    explicit aligned_storage_t( const T& x)
    {
        ::new( static_cast<void*>( &get())) T( x);
    }

    explicit aligned_storage_t( BOOST_RV_REF( T) x)
    {
        ::new( static_cast<void*>( &get())) T( boost::move( x));
    }

    ~aligned_storage_t()
    {
        get()->~T();
    }

    // Copy constructor
    aligned_storage_t( const aligned_storage_t& other)
    {
        ::new( static_cast<void*>( &get())) T( other.get());
    }

    // Copy assignment
    aligned_storage_t& operator=( aligned_storage_t other)
    {
        aligned_storage_t<T> tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    aligned_storage_t( BOOST_RV_REF( aligned_storage_t) other)
    {
        init();
        swap( other);
    }

    // Move assignment
    aligned_storage_t& operator=( BOOST_RV_REF( aligned_storage_t) other)
    {
        swap( other);
        return *this;
    }

    const T& get() const    { return *static_cast<const T*>( storage());}
    T& get()                { return *static_cast<T*>( storage());}

    void swap( aligned_storage_t& other)
    {
        boost::swap( get(), other.get());
    }

private:

    void init()
    {
        ::new( static_cast<void*>( &get())) T();
    }

    // quad word alignment
    enum { word_size = 16 };

    typedef double storage_t[ (( sizeof( T) + ( word_size - 1)) / word_size) *
                                ( word_size / sizeof( double))];
    BOOST_STATIC_ASSERT( sizeof( T) <= sizeof( storage_t));

    const void *storage() const { return &data_;}
    void *storage()             { return &data_;}

    storage_t data_;
};

template<class T>
inline void swap( aligned_storage_t<T>& x, aligned_storage_t<T>& y)
{
    x.swap( y);
}

template<class T>
class capture_allocator;

template<>
class capture_allocator<void>
{
public:

    void*               pointer;
    typedef const void* const_pointer;
    typedef void        value_type;
    template <class U> struct rebind { typedef capture_allocator<U> other;};

    friend inline bool operator==( const capture_allocator&, const capture_allocator&)
    {
        return true;
    }

    friend inline bool operator!=( const capture_allocator&, const capture_allocator&)
    {
        return false;
    }
};

template<class T>
class capture_allocator
{
public:

    typedef std::size_t         size_type;
    typedef std::ptrdiff_t      difference_type;
    typedef T*                  pointer;
    typedef const T*            const_pointer;
    typedef T&                  reference;
    typedef const T&            const_reference;
    typedef T                   value_type;
    template <typename U> struct rebind { typedef capture_allocator<U> other;};

    capture_allocator() : new_delete_( &detail::g_local_new_delete) {}

    template <typename U>
    capture_allocator( const capture_allocator<U>& x) : new_delete_( x.new_delete()) {}

    const_pointer address( const_reference x) const { return &x;}
    pointer address( reference x) const             { return &x;}

    pointer allocate( size_type n, capture_allocator<void>::const_pointer = 0)
    {
        if( n > max_size())
            throw std::bad_alloc();

        pointer result = static_cast<pointer>( new_delete_->new_( n * sizeof(T)));

        if( !result)
            throw std::bad_alloc();

        return result;
    }

    void deallocate( pointer p, size_type)
    {
        new_delete_->delete_( p);
    }

    size_type max_size() const { return size_type(-1) / sizeof(T);}

    void construct( pointer p, const T& x)
    {
        ::new( static_cast<void*>( p)) T( x);
    }

    void destroy( pointer p)
    {
        p->~T();
    }

    friend inline bool operator==( const capture_allocator& x, const capture_allocator& y)
    {
        return x.new_delete_ == y.new_delete_;
    }

    friend inline bool operator!=( const capture_allocator& x, const capture_allocator& y)
    {
        return x.new_delete_ != y.new_delete_;
    }

    const detail::new_delete_t* new_delete() const
    {
        return new_delete_;
    }

private:

    const detail::new_delete_t *new_delete_;
};

template<class T>
T *aligned_ptr( T *p, int alignment)
{
    assert( (( alignment - 1) & alignment) == 0);

    uintptr_t ptr = reinterpret_cast<uintptr_t>( p);
    uintptr_t align = alignment - 1;
    uintptr_t aligned = ( ptr + align + 1) & ~align;

    return reinterpret_cast<unsigned char *>( aligned);
}

} // core
} // ramen

#endif
