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

#ifndef RAMEN_GEO_ATTRIBUTE_TABLE_HPP
#define RAMEN_GEO_ATTRIBUTE_TABLE_HPP

#include<ramen/geo/config.hpp>

#include<boost/move/move.hpp>

#include<ramen/core/name.hpp>

#include<ramen/geo/exceptions.hpp>
#include<ramen/geo/attribute_ref.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API attribute_table_t
{
    BOOST_COPYABLE_AND_MOVABLE( attribute_table_t)

public:

    attribute_table_t();
    ~attribute_table_t();

    // Copy constructor
    attribute_table_t( const attribute_table_t& other);

    // Copy assignment
    attribute_table_t& operator=( BOOST_COPY_ASSIGN_REF( attribute_table_t) other)
    {
        attribute_table_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    attribute_table_t( BOOST_RV_REF( attribute_table_t) other)
    {
        assert( other.pimpl_);

        init();
        swap( other);
    }

    // Move assignment
    attribute_table_t& operator=( BOOST_RV_REF( attribute_table_t) other)
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    void swap( attribute_table_t& other);

    bool empty() const;
    std::size_t size() const;

    bool has_attribute( const core::name_t& name) const;
    core::type_t attribute_type( const core::name_t& name) const;

    void insert( const core::name_t& name, core::type_t attr_type);
    void insert( const core::name_t& name, const arrays::array_t& array);
    void insert( const core::name_t& name, BOOST_RV_REF( arrays::array_t) array);

    void erase( const core::name_t& name);
    void erase_arrays_indices( unsigned int start, unsigned int end);
    void erase_arrays_index( unsigned int index);

    const_attribute_ref_t get_const_attribute_ref( const core::name_t& name) const;
    attribute_ref_t get_attribute_ref( const core::name_t& name);

    const arrays::array_t& const_array( const core::name_t& name) const;
    arrays::array_t& array( const core::name_t& name);

    // iterators
    struct RAMEN_GEO_API const_iterator
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

        friend class attribute_table_t;

        explicit const_iterator( void *pimpl);

        void *pimpl_;
    };

    const_iterator begin() const;
    const_iterator end() const;

    struct RAMEN_GEO_API iterator
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

        friend class attribute_table_t;

        explicit iterator( void *pimpl);

        void *pimpl_;
    };

    iterator begin();
    iterator end();

    bool operator==( const attribute_table_t& other) const;
    bool operator!=( const attribute_table_t& other) const;

    bool check_consistency() const;

private:

    void init();

    struct impl;
    impl *pimpl_;
};

} // geo
} // ramen

#endif
