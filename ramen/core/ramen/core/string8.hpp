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

#ifndef RAMEN_CORE_STRING8_HPP
#define RAMEN_CORE_STRING8_HPP

#include<ramen/core/string_fwd.hpp>

#include<cstddef>
#include<string>
#include<iostream>

#include<boost/move/move.hpp>
#include<boost/functional/hash.hpp>

namespace ramen
{
namespace core
{

/*!
\brief UTF8 string class.
*/
class RAMEN_CORE_API string8_t
{
    BOOST_COPYABLE_AND_MOVABLE( string8_t)

public:

    typedef char                value_type;
    typedef value_type          char_type;
    typedef std::size_t         size_type;
    typedef const char_type&    const_reference;
    typedef char_type&          reference;
    typedef const char_type*    const_iterator;
    typedef char_type*          iterator;

    string8_t();

    string8_t( const char_type *str);
    string8_t( const char_type *str, std::size_t size);

    template<class Iter>
    string8_t( Iter first, Iter last)
    {
        init();
        assign( first, last);
    }

    string8_t( const string8_t& str, size_type pos, size_type n);

    // from STL string
    explicit string8_t( const std::string& s) : pimpl_( 0)
    {
        from_c_string( s.c_str(), s.size());
    }

    ~string8_t();

    // Copy constructor
    string8_t( const string8_t& other);

    // Copy assignment
    string8_t& operator=( BOOST_COPY_ASSIGN_REF( string8_t) other)
    {
        string8_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    string8_t( BOOST_RV_REF( string8_t) other) : pimpl_( 0)
    {
        assert( other.pimpl_);

        swap( other);
    }

    // Move assignment
    string8_t& operator=( BOOST_RV_REF( string8_t) other)
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    template<class Iter>
    void assign( Iter first, Iter last)
    {
        clear();
        // TODO: check if this works ok with boost tokenizer.
        //std::size_t n = std::distance( first, last);
        //reserve( size() + n);

        while( first != last)
            push_back( *first++);
    }

    void swap( string8_t& other);

    string8_t& operator=( const char *str);

    size_type size() const;

    size_type length() const;

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
    string8_t& operator+=( const char_type *str);
	string8_t& operator+=( const string8_t& str);

    void append( const char_type *str, size_type len);
    void append( const string8_t& str);

    std::string to_std_string() const
    {
        assert( pimpl_);

        return std::string( c_str());
    }

    char_type operator[]( size_type index) const;
    char_type& operator[]( size_type index);

private:

    struct impl;
    impl *pimpl_;

    void init( impl *x = 0);

    void from_c_string( const char *str, std::size_t size);
};

inline void swap( string8_t& x, string8_t& y)
{
    x.swap( y);
}

RAMEN_CORE_API string8_t operator+( const string8_t& a, const string8_t& b);
RAMEN_CORE_API string8_t operator+( const string8_t& a, const char *b);

RAMEN_CORE_API bool operator==( const string8_t& a, const string8_t& b);
RAMEN_CORE_API bool operator==( const string8_t& a, const char *b);
RAMEN_CORE_API bool operator==( const char *a, const string8_t& b);

RAMEN_CORE_API bool operator!=( const string8_t& a, const string8_t& b);
RAMEN_CORE_API bool operator!=( const string8_t& a, const char *b);
RAMEN_CORE_API bool operator!=( const char *a, const string8_t& b);

RAMEN_CORE_API bool operator<( const string8_t& a, const string8_t& b);
RAMEN_CORE_API bool operator<( const string8_t& a, const char *b);
RAMEN_CORE_API bool operator<( const char *a, const string8_t& b);

RAMEN_CORE_API const string8_t make_string( const char *a, const char *b, const char *c = 0, const char *d = 0);

inline std::ostream& operator<<( std::ostream& os, const string8_t& str)
{
    return os << str.c_str();
}

inline std::istream& operator>>( std::istream& is, string8_t& str)
{
    std::string tmp;
    is >> tmp;
    str.assign( tmp.begin(), tmp.end());
    return is;
}

inline std::size_t hash_value( const string8_t& str)
{
    return boost::hash_range( str.begin(), str.end());
}

} // core
} // ramen

#endif
