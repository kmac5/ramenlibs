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

#ifndef RAMEN_CORE_STRING16_HPP
#define RAMEN_CORE_STRING16_HPP

#include<ramen/core/config.hpp>

#include<ramen/core/string_fwd.hpp>

#include<cstddef>
#include<cassert>

#include<boost/cstdint.hpp>
#include<boost/move/move.hpp>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief string class with 16 bit characters.
*/
class RAMEN_CORE_API string16_t
{
    BOOST_COPYABLE_AND_MOVABLE( string16_t)

public:

    typedef boost::uint16_t     value_type;
    typedef value_type          char_type;
    typedef std::size_t         size_type;
    typedef const char_type&    const_reference;
    typedef char_type&          reference;
    typedef const char_type*    const_iterator;
    typedef char_type*          iterator;

    string16_t();

    string16_t( const char_type *str);
    string16_t( const char_type *str, std::size_t size);

    ~string16_t();

    // Copy constructor
    string16_t( const string16_t& other);

    // Copy assignment
    string16_t& operator=( BOOST_COPY_ASSIGN_REF( string16_t) other)
    {
        string16_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    string16_t( BOOST_RV_REF( string16_t) other) : pimpl_( 0)
    {
        assert( other.pimpl_);

        swap( other);
    }

    // Move assignment
    string16_t& operator=( BOOST_RV_REF( string16_t) other)
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    void swap( string16_t& other);

    string16_t& operator=( const char_type *str);

    size_type size() const;

    bool empty() const;

    void reserve( size_type n);

    void clear();

    const char_type *c_str() const;

    void push_back( char_type c);

    const_iterator begin() const;
    const_iterator end() const;

    iterator begin();
    iterator end();

    // append
    string16_t& operator+=( const char_type *str);
    string16_t& operator+=( const string16_t& str);

    void append( const char_type *str, size_type len);
    void append( const string16_t& str);

private:

    struct impl;
    impl *pimpl_;

    void init( impl *x = 0);
};

inline void swap( string16_t& x, string16_t& y)
{
    x.swap( y);
}

bool RAMEN_CORE_API operator==( const string16_t& a, const string16_t& b);
bool RAMEN_CORE_API operator!=( const string16_t& a, const string16_t& b);
bool RAMEN_CORE_API operator<( const string16_t& a, const string16_t& b);

} // core
} // ramen

#endif
