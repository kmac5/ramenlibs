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

#include<ramen/containers/dictionary.hpp>

#include<algorithm>

#include<boost/container/flat_map.hpp>

#include<ramen/core/exceptions.hpp>
#include<ramen/core/new_allocator.hpp>
#include<ramen/core/stl_allocator_adapter.hpp>

namespace ramen
{
namespace containers
{

struct dictionary_t::impl
{
    typedef std::pair<key_type, value_type>             pair_type;
    typedef core::stl_allocator_adapter_t<pair_type>    allocator_type;
    typedef std::less<key_type>                         compare_type;

    typedef boost::container::flat_map< key_type,
                                        value_type,
                                        std::less<key_type>,
                                        allocator_type> map_type;

    typedef map_type::const_iterator                    const_iterator;
    typedef map_type::iterator                          iterator;

    impl() : items( compare_type(), allocator_type( core::global_new_allocator()))
    {
    }

    explicit impl( const core::allocator_ptr_t& alloc) : items( compare_type(), allocator_type( alloc))
    {
    }

    map_type items;
};

dictionary_t::dictionary_t() : pimpl_( 0)
{
}

dictionary_t::dictionary_t( const core::allocator_ptr_t& alloc) : pimpl_( 0)
{
    assert( alloc);

    pimpl_ = new impl( alloc);
}

dictionary_t::~dictionary_t()
{
    delete pimpl_;
}

// Copy constructor
dictionary_t::dictionary_t( const dictionary_t& other) : pimpl_( 0)
{
    if( other.pimpl_)
        pimpl_ = new impl( *other.pimpl_);
}

void dictionary_t::swap( dictionary_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

bool dictionary_t::empty() const
{
    if( pimpl_)
        return pimpl_->items.empty();

    return true;
}

dictionary_t::size_type dictionary_t::size() const
{
    if( pimpl_)
        return pimpl_->items.size();

    return 0;
}

void dictionary_t::clear()
{
    if( pimpl_)
        pimpl_->items.clear();
}

const dictionary_t::value_type& dictionary_t::operator[]( const dictionary_t::key_type& key) const
{
    if( !pimpl_)
        throw core::key_not_found( key);

    impl::const_iterator it( pimpl_->items.find( key ));

    if( it != pimpl_->items.end())
        return it->second;
    else
        throw core::key_not_found( key);
}

dictionary_t::value_type& dictionary_t::operator[]( const dictionary_t::key_type& key)
{
    if( !pimpl_)
        pimpl_ = new impl();

    impl::iterator it( pimpl_->items.find( key ));

    if( it != pimpl_->items.end())
        return it->second;
    else
    {
        std::pair<impl::iterator, bool> new_it( pimpl_->items.insert( std::make_pair( key,
                                                                                      dictionary_t::value_type())));
        return new_it.first->second;
    }
}

dictionary_t::const_iterator dictionary_t::begin() const
{
    if( pimpl_)
        return reinterpret_cast<const_iterator>( pimpl_->items.begin().get_ptr());

    return 0;
}

dictionary_t::const_iterator dictionary_t::end() const
{
    if( pimpl_)
        return reinterpret_cast<const_iterator>( pimpl_->items.end().get_ptr());

    return 0;
}

dictionary_t::iterator dictionary_t::begin()
{
    if( pimpl_)
        return reinterpret_cast<iterator>( pimpl_->items.begin().get_ptr());

    return 0;
}

dictionary_t::iterator dictionary_t::end()
{
    if( pimpl_)
        return reinterpret_cast<iterator>( pimpl_->items.end().get_ptr());

    return 0;
}

bool dictionary_t::operator==( const dictionary_t& other) const
{
    if( pimpl_ == 0 && other.pimpl_ == 0)
        return true;

    if( pimpl_ != 0 && other.pimpl_ != 0)
        return pimpl_->items == other.pimpl_->items;

    return false;
}

bool dictionary_t::operator!=( const dictionary_t& other) const
{
    return !( *this == other);
}

const dictionary_t::value_type *dictionary_t::get( const dictionary_t::key_type& key) const
{
    if( !pimpl_)
        return 0;

    impl::const_iterator it( pimpl_->items.find( key ));

    if( it != pimpl_->items.end())
        return &( it->second);

    return 0;
}

dictionary_t::value_type *dictionary_t::get( const dictionary_t::key_type& key)
{
    if( !pimpl_)
        return 0;

    impl::iterator it( pimpl_->items.find( key ));

    if( it != pimpl_->items.end())
        return &( it->second);

    return 0;
}

} // containers
} // ramen
