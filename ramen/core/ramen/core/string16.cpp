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

#include<ramen/core/string16.hpp>

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

namespace
{

boost::uint16_t empty_str = 0;

} // unnamed

struct string16_t::impl
{
    boost::container::vector<string16_t::char_type> str_;
};

string16_t::string16_t() : pimpl_( 0)
{
    init();
}

string16_t::string16_t( const char_type *str) : pimpl_( 0)
{
    init();
    std::size_t size = detail::string_length( str);
    pimpl_->str_.reserve( size + 1);
    pimpl_->str_.assign( str, str + size);
    pimpl_->str_.push_back( char_type( 0));
}

string16_t::string16_t( const char_type *str, std::size_t size) : pimpl_( 0)
{
    init();
    pimpl_->str_.reserve( size + 1);
    pimpl_->str_.assign( str, str + size);
    pimpl_->str_.push_back( char_type( 0));
}

void string16_t::init( impl *x)
{
    assert( !pimpl_);

    if( x)
        pimpl_ = new impl( *x);
    else
        pimpl_ = new impl();
}

string16_t::~string16_t()
{
    delete pimpl_;
}

// Copy constructor
string16_t::string16_t( const string16_t& other) : pimpl_( 0)
{
    assert( other.pimpl_);

    init( other.pimpl_);
}

string16_t& string16_t::operator=( const char_type *str)
{
    string16_t tmp( str);
    swap( tmp);
    return *this;
}

string16_t::size_type string16_t::size() const
{
    assert( pimpl_);

    return pimpl_->str_.empty() ? 0 : pimpl_->str_.size() - 1;
}

bool string16_t::empty() const
{
    assert( pimpl_);

    return pimpl_->str_.empty();
}

void string16_t::reserve( size_type n)
{
    assert( pimpl_);

    pimpl_->str_.reserve( n == 0 ? 0 : n + 1);
}

void string16_t::clear()
{
    assert( pimpl_);

    pimpl_->str_.clear();
}

const string16_t::char_type *string16_t::c_str() const
{
    assert( pimpl_);

    return pimpl_->str_.empty() ? &empty_str : &(pimpl_->str_[0]);
}

void string16_t::push_back( char_type c)
{
    assert( pimpl_);

    if( !pimpl_->str_.empty())
        pimpl_->str_.pop_back();

    pimpl_->str_.push_back( c);
    pimpl_->str_.push_back( char_type( 0));
}

string16_t::const_iterator string16_t::begin() const
{
    assert( pimpl_);

    return pimpl_->str_.begin().get_ptr();
}

string16_t::const_iterator string16_t::end() const
{
    assert( pimpl_);

    if( !pimpl_->str_.empty())
        return pimpl_->str_.end().get_ptr() - 1;

    return pimpl_->str_.end().get_ptr();
}

string16_t::iterator string16_t::begin()
{
    assert( pimpl_);

    return pimpl_->str_.begin().get_ptr();
}

string16_t::iterator string16_t::end()
{
    assert( pimpl_);

    if( !pimpl_->str_.empty())
        return pimpl_->str_.end().get_ptr() - 1;

    return pimpl_->str_.end().get_ptr();
}

string16_t& string16_t::operator+=( const char_type *str)
{
    append( str, detail::string_length( str));
    return *this;
}

string16_t& string16_t::operator+=( const string16_t& str)
{
    append( str);
    return *this;
}

void string16_t::append( const char_type *str, size_type len)
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

void string16_t::append( const string16_t& str)
{
    append( str.c_str(), str.size());
}

void string16_t::swap( string16_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

bool operator==( const string16_t& a, const string16_t& b)
{
    return detail::string_compare( a.c_str(), b.c_str()) == 0;
}

bool operator!=( const string16_t& a, const string16_t& b)
{
    return !( a == b);
}

bool operator<( const string16_t& a, const string16_t& b)
{
    return detail::string_compare( a.c_str(), b.c_str()) < 0;
}

} // core
} // ramen
