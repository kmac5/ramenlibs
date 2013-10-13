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

#ifndef RAMEN_ARRAYS_NAMED_ARRAY_MAP_HPP
#define RAMEN_ARRAYS_NAMED_ARRAY_MAP_HPP

#include<ramen/arrays/config.hpp>

#include<boost/move/move.hpp>

#include<ramen/core/name.hpp>
#include<ramen/arrays/array.hpp>
#include<ramen/arrays/apply_visitor.hpp>

namespace ramen
{
namespace arrays
{

class RAMEN_ARRAYS_API named_array_map_t
{
    BOOST_COPYABLE_AND_MOVABLE( named_array_map_t)
    
public:
    
    named_array_map_t();
    ~named_array_map_t();

    // Copy constructor
    named_array_map_t( const named_array_map_t& other);

    // Copy assignment
    named_array_map_t& operator=( BOOST_COPY_ASSIGN_REF( named_array_map_t) other)
    {
        named_array_map_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    named_array_map_t( BOOST_RV_REF( named_array_map_t) other) : pimpl_( 0)
    {
        swap( other);
    }

    // Move assignment
    named_array_map_t& operator=( BOOST_RV_REF( named_array_map_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( named_array_map_t& other);
    
    bool empty() const;
    std::size_t size() const;
    void clear();
    
    bool has_array( const core::name_t& name) const;
    
    void insert( const core::name_t& name, core::type_t attr_type);
    void insert( const core::name_t& name, const arrays::array_t& array);
    void insert( const core::name_t& name, BOOST_RV_REF( arrays::array_t) array);

    void erase( const core::name_t& name);
    
    const array_t& array( const core::name_t& name) const;
    array_t& array( const core::name_t& name);
    
    // iterators
    struct RAMEN_ARRAYS_API const_iterator
    {
        const_iterator();

        const_iterator& operator++();
        const_iterator operator++( int);

        const_iterator& operator--();
        const_iterator operator--( int);

        const core::name_t& first() const;
        const arrays::array_t& second() const;

        bool operator==( const const_iterator& other) const;
        bool operator!=( const const_iterator& other) const;

    private:

        friend class named_array_map_t;

        explicit const_iterator( void *pimpl);

        void *pimpl_;
    };

    const_iterator begin() const;
    const_iterator end() const;

    struct RAMEN_ARRAYS_API iterator
    {
        iterator();

        iterator& operator++();
        iterator operator++( int);

        iterator& operator--();
        iterator operator--( int);

        const core::name_t& first() const;
        arrays::array_t& second();

        bool operator==( const iterator& other) const;
        bool operator!=( const iterator& other) const;

    private:

        friend class named_array_map_t;

        explicit iterator( void *pimpl);

        void *pimpl_;
    };

    iterator begin();
    iterator end();
    
    bool operator==( const named_array_map_t& other) const;
    bool operator!=( const named_array_map_t& other) const;
    
    template<class Visitor>
    void apply_visitor( Visitor& v) const
    {
        for( const_iterator it( begin()), e( end()); it != e; ++it)
            ramen::arrays::apply_visitor( v, it.second());
    }

    template<class Visitor>
    void apply_visitor( Visitor& v)
    {
        for( iterator it( begin()), e( end()); it != e; ++it)
            ramen::arrays::apply_visitor( v, it.second());
    }

    template<class Visitor, class Predicate>
    void apply_visitor_if( Visitor& v, Predicate pred) const
    {
        for( const_iterator it( begin()), e( end()); it != e; ++it)
            if( pred( it.first()))
                ramen::arrays::apply_visitor( v, it.second());
    }

    template<class Visitor, class Predicate>
    void apply_visitor_if( Visitor& v, Predicate pred)
    {
        for( iterator it( begin()), e( end()); it != e; ++it)
            if( pred( it.first()))
                ramen::arrays::apply_visitor( v, it.second());
    }

private:

    struct impl;
    impl *pimpl_;
};

inline void swap( named_array_map_t& x, named_array_map_t& y)
{
    x.swap( y);
}

} // arrays
} // ramen

#endif
