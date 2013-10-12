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

#include<ramen/arrays/named_array_map.hpp>

#include<boost/container/flat_map.hpp>

#include<ramen/core/exceptions.hpp>

namespace ramen
{
namespace arrays
{

struct named_array_map_t::impl
{
    typedef boost::container::flat_map<core::name_t, arrays::array_t>   map_type;
    typedef std::pair<core::name_t, arrays::array_t >                   pair_type;

    typedef map_type::const_iterator    const_iterator;
    typedef map_type::iterator          iterator;

    map_type map;
};

named_array_map_t::named_array_map_t()
{
    pimpl_ = new impl();
}

named_array_map_t::~named_array_map_t()
{
    delete pimpl_;
}

named_array_map_t::named_array_map_t( const named_array_map_t& other)
{
    pimpl_ = new impl( *other.pimpl_);
}

void named_array_map_t::swap( named_array_map_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

bool named_array_map_t::empty() const
{
    assert( pimpl_);

    return pimpl_->map.empty();
}

std::size_t named_array_map_t::size() const
{
    assert( pimpl_);

    return pimpl_->map.size();
}

void named_array_map_t::clear()
{
    assert( pimpl_);

    pimpl_->map.clear();
}

bool named_array_map_t::has_array( const core::name_t& name) const
{
    assert( pimpl_);

    if( pimpl_->map.find( name) != pimpl_->map.end())
        return true;

    return false;
}

void named_array_map_t::insert( const core::name_t& name, core::type_t attr_type)
{
    pimpl_->map[name] = arrays::array_t( attr_type);
}

void named_array_map_t::insert( const core::name_t& name, const arrays::array_t& array)
{
    pimpl_->map[name] = array;
}

void named_array_map_t::insert( const core::name_t& name, BOOST_RV_REF( arrays::array_t) array)
{
    pimpl_->map[name] = array;
}

void named_array_map_t::erase( const core::name_t& name)
{
    assert( pimpl_);

    pimpl_->map.erase( name);
}

const array_t& named_array_map_t::const_array( const core::name_t& name) const
{
    assert( pimpl_);   

    impl::const_iterator it( pimpl_->map.find( name));
    
    if( it == pimpl_->map.end())
        throw core::key_not_found( name);

    return it->second;
}

array_t& named_array_map_t::array( const core::name_t& name)
{
    assert( pimpl_);   

    impl::iterator it( pimpl_->map.find( name));

    if( it == pimpl_->map.end())
        throw core::key_not_found( name);

    return it->second;
}

// iterators

named_array_map_t::const_iterator::const_iterator() : pimpl_( 0) {}

named_array_map_t::const_iterator::const_iterator( void *pimpl) : pimpl_( pimpl) {}

named_array_map_t::const_iterator& named_array_map_t::const_iterator::operator++()
{
    typedef named_array_map_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) + 1);
    return *this;
}

named_array_map_t::const_iterator named_array_map_t::const_iterator::operator++( int)
{
    const_iterator saved( *this);
    ++( *this);
    return saved;
}

named_array_map_t::const_iterator& named_array_map_t::const_iterator::operator--()
{
    typedef named_array_map_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) - 1);
    return *this;
}

named_array_map_t::const_iterator named_array_map_t::const_iterator::operator--( int)
{
    const_iterator saved( *this);
    --( *this);
    return saved;
}

const core::name_t& named_array_map_t::const_iterator::first() const
{
    assert( pimpl_);

    typedef named_array_map_t::impl::pair_type pair_type;
    return reinterpret_cast<const pair_type*>( pimpl_)->first;
}

const arrays::array_t& named_array_map_t::const_iterator::second() const
{
    assert( pimpl_);

    typedef named_array_map_t::impl::pair_type pair_type;
    return reinterpret_cast<const pair_type*>( pimpl_)->second;
}

bool named_array_map_t::const_iterator::operator==( const const_iterator& other) const
{
    return pimpl_ == other.pimpl_;
}

bool named_array_map_t::const_iterator::operator!=( const const_iterator& other) const
{
    return !( *this == other);
}

named_array_map_t::const_iterator named_array_map_t::begin() const
{
    assert( pimpl_);

    return const_iterator( pimpl_->map.begin().get_ptr());
}

named_array_map_t::const_iterator named_array_map_t::end() const
{
    assert( pimpl_);

    return const_iterator( pimpl_->map.end().get_ptr());
}

named_array_map_t::iterator::iterator() : pimpl_( 0) {}

named_array_map_t::iterator::iterator( void *pimpl) : pimpl_( pimpl) {}

named_array_map_t::iterator& named_array_map_t::iterator::operator++()
{
    typedef named_array_map_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) + 1);
    return *this;
}

named_array_map_t::iterator named_array_map_t::iterator::operator++( int)
{
    iterator saved( *this);
    ++( *this);
    return saved;
}

named_array_map_t::iterator& named_array_map_t::iterator::operator--()
{
    typedef named_array_map_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) - 1);
    return *this;
}

named_array_map_t::iterator named_array_map_t::iterator::operator--( int)
{
    iterator saved( *this);
    --( *this);
    return saved;
}

const core::name_t& named_array_map_t::iterator::first() const
{
    assert( pimpl_);

    typedef named_array_map_t::impl::pair_type pair_type;
    return reinterpret_cast<const pair_type*>( pimpl_)->first;
}

arrays::array_t& named_array_map_t::iterator::second()
{
    assert( pimpl_);

    typedef named_array_map_t::impl::pair_type pair_type;
    return reinterpret_cast<pair_type*>( pimpl_)->second;
}

bool named_array_map_t::iterator::operator==( const iterator& other) const
{
    return pimpl_ == other.pimpl_;
}

bool named_array_map_t::iterator::operator!=( const iterator& other) const
{
    return !( *this == other);
}

named_array_map_t::iterator named_array_map_t::begin()
{
    assert( pimpl_);

    return iterator( pimpl_->map.begin().get_ptr());
}

named_array_map_t::iterator named_array_map_t::end()
{
    assert( pimpl_);

    return iterator( pimpl_->map.end().get_ptr());
}

bool named_array_map_t::operator==( const named_array_map_t& other) const
{
    assert( pimpl_);
    assert( other.pimpl_);

    return pimpl_->map == other.pimpl_->map;
}

bool named_array_map_t::operator!=( const named_array_map_t& other) const
{
    return !( *this == other);
}

} // arrays
} // ramen
