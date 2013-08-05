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

#ifndef RAMEN_CORE_STRING8_VECTOR_HPP
#define RAMEN_CORE_STRING8_VECTOR_HPP

#include<ramen/core/string8_vector_fwd.hpp>

#include<iterator>

#include<boost/move/move.hpp>

#include<ramen/core/string_fwd.hpp>

#include<ramen/core/allocator_interface.hpp>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief A vector of unique string8_t strings.
*/
class RAMEN_CORE_API string8_vector_t
{
    BOOST_COPYABLE_AND_MOVABLE( string8_vector_t)

public:

    typedef core::string8_t value_type;
    typedef std::size_t     size_type;

    string8_vector_t();

    explicit string8_vector_t( const allocator_ptr_t& alloc);

    ~string8_vector_t();

    // Copy constructor
    string8_vector_t( const string8_vector_t& other);

    // Copy assignment
    string8_vector_t& operator=( BOOST_COPY_ASSIGN_REF( string8_vector_t) other)
    {
        string8_vector_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    string8_vector_t( BOOST_RV_REF( string8_vector_t) other) : pimpl_( 0)
    {
        swap( other);
    }

    // Move assignment
    string8_vector_t& operator=( BOOST_RV_REF( string8_vector_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( string8_vector_t& other);

    bool operator==( const string8_vector_t& other) const;
    bool operator!=( const string8_vector_t& other) const;

    bool empty() const;
    size_type size() const;
    size_type capacity() const;

    void clear();

    void reserve( size_type n);
    void shrink_to_fit();
    void resize( size_type n);

    void push_back( const string8_t& str);
    void erase( size_type start, size_type end);

    // iterators and proxies...

    const string8_t& operator[]( std::size_t n) const;

    class const_iterator
    {
    public:

        typedef std::random_access_iterator_tag iterator_category;
        typedef const core::string8_t           value_type;
        typedef std::ptrdiff_t                  difference_type;
        typedef value_type*                     pointer_type;
        typedef value_type&                     reference_type;

        typedef reference_type    reference;
        typedef pointer_type      pointer;

        reference_type operator*() const;
        reference_type operator->() const;
        reference_type operator[]( int n) const;

        const_iterator& operator++();
        const_iterator  operator++( int);

        const_iterator& operator+=( int n);
        const_iterator& operator-=( int n);

        bool operator<( const const_iterator& other) const;
        bool operator==( const const_iterator& other) const;
        bool operator!=( const const_iterator& other) const;

        difference_type operator-( const const_iterator& other) const;

    private:

        friend class string8_vector_t;

        const_iterator( const string8_vector_t *parent, int index);

        const string8_vector_t *parent_;
        int index_;
    };

    const_iterator begin() const;
    const_iterator end() const;

    class string_proxy
    {
    public:

        operator const string8_t&() const;

        string_proxy& operator=( const core::string8_t& str);

    private:

        friend class string8_vector_t;

        string_proxy( string8_vector_t *parent, int index);

        string8_vector_t *parent_;
        int index_;
    };

    string_proxy operator[]( std::size_t n);

    class iterator
    {
    public:

        typedef std::random_access_iterator_tag iterator_category;
        typedef core::string8_t                 value_type;
        typedef std::ptrdiff_t                  difference_type;
        typedef string_proxy                    reference_type;
        typedef reference_type*                 pointer_type;

        typedef string_proxy    reference;
        typedef reference*      pointer;

        reference_type operator*() const;
        reference_type operator->() const;
        reference_type operator[]( int n) const;

        iterator& operator++();
        iterator  operator++( int);

        iterator& operator+=( int n);
        iterator& operator-=( int n);

        bool operator<( const iterator& other) const;
        bool operator==( const iterator& other) const;
        bool operator!=( const iterator& other) const;

        difference_type operator-( const iterator& other) const;

    private:

        friend class string8_vector_t;

        iterator( string8_vector_t *parent, int index);

        string8_vector_t *parent_;
        int index_;
    };

    iterator begin();
    iterator end();

private:

    friend class string_proxy;

    struct impl;
    impl *pimpl_;
};

inline void swap( string8_vector_t& x, string8_vector_t& y)
{
    x.swap( y);
}

} // core
} // ramen

#endif
