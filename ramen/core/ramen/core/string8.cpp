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

#include<ramen/core/string8.hpp>

#include<stdlib.h>
#include<string.h>

#include<algorithm>
#include<cstring>

#include<boost/container/vector.hpp>

#include<ramen/core/memory.hpp>
#include<ramen/core/exceptions.hpp>
#include<ramen/core/detail/string.hpp>

namespace ramen
{
namespace core
{

struct string8_t::impl
{
    boost::container::vector<string8_t::char_type> str_;
};

string8_t::string8_t() : pimpl_( 0)
{
    init();
}

string8_t::string8_t( const char_type *str) : pimpl_( 0)
{
    from_c_string( str, detail::string_length( str));
}

string8_t::string8_t( const char_type *str, std::size_t size) : pimpl_( 0)
{
    from_c_string( str, size);
}

string8_t::string8_t( const string8_t& str, size_type pos, size_type n) : pimpl_( 0)
{
    init();
    size_type nn = std::min( n, str.length() - pos);
    assign( str.begin() + pos, str.begin() + pos + nn);
}

void string8_t::from_c_string( const char *str, std::size_t size)
{
    init();
    pimpl_->str_.reserve( size + 1);
    pimpl_->str_.assign( str, str + size);
    pimpl_->str_.push_back( char_type( 0));
}

void string8_t::init( impl *x)
{
    assert( !pimpl_);

    if( x)
        pimpl_ = new impl( *x);
    else
        pimpl_ = new impl();
}

string8_t::~string8_t()
{
    delete pimpl_;
}

string8_t::string8_t( const string8_t& other) : pimpl_( 0)
{
    assert( other.pimpl_);

    init( other.pimpl_);
}

void string8_t::swap( string8_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

string8_t& string8_t::operator=( const char_type *str)
{
    string8_t tmp( str);
    swap( tmp);
    return *this;
}

string8_t::size_type string8_t::size() const
{
    assert( pimpl_);

    return pimpl_->str_.empty() ? 0 : pimpl_->str_.size() - 1;
}

string8_t::size_type string8_t::length() const
{
    return size();
}

bool string8_t::empty() const
{
    assert( pimpl_);

    return pimpl_->str_.empty();
}

void string8_t::reserve( size_type n)
{
    assert( pimpl_);

    pimpl_->str_.reserve( n == 0 ? 0 : n + 1);
}

void string8_t::clear()
{
    assert( pimpl_);

    pimpl_->str_.clear();
}

const string8_t::char_type *string8_t::c_str() const
{
    assert( pimpl_);

    return pimpl_->str_.empty() ? "" : &(pimpl_->str_[0]);
}

void string8_t::push_back( char_type c)
{
    assert( pimpl_);

    if( !pimpl_->str_.empty())
        pimpl_->str_.pop_back();

    pimpl_->str_.push_back( c);
    pimpl_->str_.push_back( char_type( 0));
}

string8_t::const_iterator string8_t::begin() const
{
    assert( pimpl_);

    return pimpl_->str_.begin().get_ptr();
}

string8_t::const_iterator string8_t::end() const
{
    assert( pimpl_);

    if( !pimpl_->str_.empty())
        return pimpl_->str_.end().get_ptr() - 1;

    return pimpl_->str_.end().get_ptr();
}

string8_t::iterator string8_t::begin()
{
    assert( pimpl_);

    return pimpl_->str_.begin().get_ptr();
}

string8_t::iterator string8_t::end()
{
    assert( pimpl_);

    if( !pimpl_->str_.empty())
        return pimpl_->str_.end().get_ptr() - 1;

    return pimpl_->str_.end().get_ptr();
}

string8_t& string8_t::operator+=( const char_type *str)
{
    append( str, detail::string_length( str));
    return *this;
}

string8_t& string8_t::operator+=( const string8_t& str)
{
    append( str);
    return *this;
}

void string8_t::append( const char_type *str, size_type len)
{
    assert( pimpl_);

    if( len != 0)
    {
        if( !pimpl_->str_.empty())
            pimpl_->str_.pop_back();

        pimpl_->str_.insert( pimpl_->str_.end(), str, str + len);
        pimpl_->str_.push_back(0);
    }
}

void string8_t::append( const string8_t& str)
{
    append( str.c_str(), str.size());
}

string8_t::char_type string8_t::operator[]( size_type index) const
{
    assert( pimpl_);
    assert( index < size());

    return pimpl_->str_[index];
}

string8_t::char_type& string8_t::operator[]( size_type index)
{
    assert( pimpl_);
    assert( index < size());

    return pimpl_->str_[index];
}

string8_t operator+( const string8_t& a, const string8_t& b)
{
    string8_t result( a);
    result.append( b);
    return result;
}

string8_t operator+( const string8_t& a, const char *b)
{
    string8_t result( a);
    result.append( b, detail::string_length( b));
    return result;
}

bool operator==( const string8_t& a, const string8_t& b)
{
    return detail::string_compare( a.c_str(), b.c_str()) == 0;
}

bool operator==( const string8_t& a, const char *b)
{
    return detail::string_compare( a.c_str(), b) == 0;
}

bool operator==( const char *a, const string8_t& b)
{
    return detail::string_compare( a, b.c_str()) == 0;
}

bool operator!=( const string8_t& a, const string8_t& b)
{
    return !( a == b);
}

bool operator!=( const string8_t& a, const char *b)
{
    return !( a == b);
}

bool operator!=( const char *a, const string8_t& b)
{
    return !( a == b);
}

bool operator<( const string8_t& a, const string8_t& b)
{
    return detail::string_compare( a.c_str(), b.c_str()) < 0;
}

bool operator<( const string8_t& a, const char *b)
{
    return detail::string_compare( a.c_str(), b) < 0;
}

bool operator<( const char *a, const string8_t& b)
{
    return detail::string_compare( a, b.c_str()) < 0;
}

const string8_t make_string( const char *a, const char *b, const char *c, const char *d)
{
    assert( a);
    assert( b);

    std::size_t a_len = detail::string_length( a);
    std::size_t b_len = detail::string_length( b);
    std::size_t c_len = ( c != 0) ? detail::string_length( c) : 0;
    std::size_t d_len = ( d != 0) ? detail::string_length( d) : 0;

    string8_t str;
    str.reserve( a_len + b_len + c_len + d_len + 1);
    str.append( a, a_len);
    str.append( b, b_len);

    if( c)
        str.append( c, c_len);

    if( d)
        str.append( d, d_len);

    return str;
}

} // core
} // ramen
