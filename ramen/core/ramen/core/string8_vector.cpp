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

#include<ramen/core/string8_vector.hpp>

#include<algorithm>

#include<boost/container/vector.hpp>
#include<boost/flyweight.hpp>

#include<ramen/core/exceptions.hpp>

namespace ramen
{
namespace core
{

struct string8_vector_t::impl
{
    bool operator==( const string8_vector_t::impl& other) const
    {
        return items == other.items;
    }

    boost::container::vector<boost::flyweights::flyweight<core::string8_t> > items;
};

string8_vector_t::string8_vector_t() : pimpl_( 0) {}

string8_vector_t::string8_vector_t( const allocator_ptr_t& alloc)
{
    assert( alloc);

    // TODO: implement this...
    assert( false);
}

string8_vector_t::~string8_vector_t()
{
    delete pimpl_;
}

// Copy constructor
string8_vector_t::string8_vector_t( const string8_vector_t& other) : pimpl_( 0)
{
    if( other.pimpl_)
        pimpl_ = new impl( *other.pimpl_);
}

void string8_vector_t::swap( string8_vector_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

bool string8_vector_t::empty() const
{
    if( pimpl_)
        return pimpl_->items.empty();

    return true;
}

string8_vector_t::size_type string8_vector_t::size() const
{
    if( pimpl_)
        return pimpl_->items.size();

    return 0;
}

string8_vector_t::size_type string8_vector_t::capacity() const
{
    if( pimpl_)
        return pimpl_->items.capacity();

    return 0;
}

void string8_vector_t::clear()
{
    if( pimpl_)
        pimpl_->items.clear();
}

void string8_vector_t::reserve( size_type n)
{
    if( !pimpl_)
        pimpl_ = new impl;

    pimpl_->items.reserve( n);
}

void string8_vector_t::shrink_to_fit()
{
    if( !pimpl_)
        pimpl_->items.shrink_to_fit();
}

void string8_vector_t::resize( size_type n)
{
    assert( pimpl_);

    pimpl_->items.resize( n);
}

void string8_vector_t::push_back( const string8_t& str)
{
    if( !pimpl_)
        pimpl_ = new impl;

    pimpl_->items.push_back( str);
}

void string8_vector_t::erase( size_type start, size_type end)
{
    assert( pimpl_);

    pimpl_->items.erase( pimpl_->items.begin() + start,
                         pimpl_->items.begin() + end);
}

const string8_t& string8_vector_t::operator[]( std::size_t n) const
{
    assert( pimpl_);
    assert( n < size());

    return pimpl_->items[n];
}

bool string8_vector_t::operator==( const string8_vector_t& other) const
{
    if( pimpl_ == 0 && other.pimpl_ == 0)
        return true;

    if( pimpl_ != 0 && other.pimpl_ != 0)
        return *pimpl_ == *other.pimpl_;

    return false;
}

bool string8_vector_t::operator!=( const string8_vector_t& other) const
{
    return !( *this == other);
}

// const iterator

string8_vector_t::const_iterator::const_iterator( const string8_vector_t *parent,
                                                  int index) : parent_( parent), index_( index)
{
    assert( parent_);
    assert( index_ >= 0);
}

string8_vector_t::const_iterator::reference_type string8_vector_t::const_iterator::operator*() const
{
    return ( *parent_)[index_];
}

string8_vector_t::const_iterator::reference_type string8_vector_t::const_iterator::operator->() const
{
    return ( *parent_)[index_];
}

string8_vector_t::const_iterator::reference_type string8_vector_t::const_iterator::operator[]( int n) const
{
    return ( *parent_)[index_ + n];
}

string8_vector_t::const_iterator& string8_vector_t::const_iterator::operator++()
{
    ++index_;
    return *this;
}

string8_vector_t::const_iterator string8_vector_t::const_iterator::operator++( int)
{
    const_iterator tmp( *this);
    ++tmp;
    return tmp;
}

string8_vector_t::const_iterator& string8_vector_t::const_iterator::operator+=( int n)
{
    assert( n >= 0);

    index_ += n;
    return *this;
}

string8_vector_t::const_iterator& string8_vector_t::const_iterator::operator-=( int n)
{
    assert( n >= 0);

    index_ -= n;
    return *this;
}

bool string8_vector_t::const_iterator::operator<( const const_iterator& other) const
{
    return index_ < other.index_;
}

bool string8_vector_t::const_iterator::operator==( const const_iterator& other) const
{
    return index_ == other.index_;
}

bool string8_vector_t::const_iterator::operator!=( const const_iterator& other) const
{
    return index_ != other.index_;
}

string8_vector_t::const_iterator::difference_type string8_vector_t::const_iterator::operator-( const string8_vector_t::const_iterator& other) const
{
    return index_ - other.index_;
}

string8_vector_t::const_iterator string8_vector_t::begin() const
{
    if( pimpl_)
        return const_iterator( this, 0);

    return const_iterator( this, 0);
}

string8_vector_t::const_iterator string8_vector_t::end() const
{
    if( pimpl_)
        return const_iterator( this, size() + 1);

    return const_iterator( this, 0);
}

// non const iterator

string8_vector_t::string_proxy::string_proxy( string8_vector_t *parent, int index) : parent_( parent), index_( index)
{
    assert( parent_);
    assert( index_ >= 0);
}

string8_vector_t::string_proxy::operator const string8_t&() const
{
    assert( parent_->pimpl_);
    assert( index_ < parent_->size());

    return parent_->pimpl_->items[index_];
}

string8_vector_t::string_proxy& string8_vector_t::string_proxy::operator=( const core::string8_t& str)
{
    assert( parent_->pimpl_);
    assert( index_ < parent_->size());

    parent_->pimpl_->items[index_] = str;
    return *this;
}

string8_vector_t::string_proxy string8_vector_t::operator[]( std::size_t n)
{
    assert( pimpl_);
    assert( n < size());

    return string_proxy( this, n);
}

string8_vector_t::iterator::iterator( string8_vector_t *parent,
                                      int index) : parent_( parent), index_( index)
{
    assert( parent_);
    assert( index_ >= 0);
}

string8_vector_t::iterator::reference_type string8_vector_t::iterator::operator*() const
{
    return ( *parent_)[index_];
}

string8_vector_t::iterator::reference_type string8_vector_t::iterator::operator->() const
{
    return ( *parent_)[index_];
}

string8_vector_t::iterator::reference_type string8_vector_t::iterator::operator[]( int n) const
{
    return ( *parent_)[index_ + n];
}

string8_vector_t::iterator& string8_vector_t::iterator::operator++()
{
    ++index_;
    return *this;
}

string8_vector_t::iterator string8_vector_t::iterator::operator++( int)
{
    iterator tmp( *this);
    ++tmp;
    return tmp;
}

string8_vector_t::iterator& string8_vector_t::iterator::operator+=( int n)
{
    assert( n >= 0);

    index_ += n;
    return *this;
}

string8_vector_t::iterator& string8_vector_t::iterator::operator-=( int n)
{
    assert( n >= 0);

    index_ -= n;
    return *this;
}

bool string8_vector_t::iterator::operator<( const iterator& other) const
{
    return index_ < other.index_;
}

bool string8_vector_t::iterator::operator==( const iterator& other) const
{
    return index_ == other.index_;
}

bool string8_vector_t::iterator::operator!=( const iterator& other) const
{
    return index_ != other.index_;
}

string8_vector_t::iterator::difference_type string8_vector_t::iterator::operator-( const string8_vector_t::iterator& other) const
{
    return index_ - other.index_;
}

string8_vector_t::iterator string8_vector_t::begin()
{
    if( pimpl_)
        return iterator( this, 0);

    return iterator( this, 0);
}

string8_vector_t::iterator string8_vector_t::end()
{
    if( pimpl_)
        return iterator( this, size() + 1);

    return iterator( this, 0);
}

} // core
} // ramen
