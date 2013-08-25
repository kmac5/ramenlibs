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

#include<ramen/geo/attribute_table.hpp>

#include<algorithm>

#include<boost/container/flat_map.hpp>

#include<ramen/core/memory.hpp>
#include<ramen/core/copy_on_write.hpp>

#include<ramen/arrays/apply_visitor.hpp>

#include<ramen/geo/exceptions.hpp>

namespace ramen
{
namespace geo
{

struct attribute_table_t::impl
{
    typedef boost::container::flat_map<core::name_t, core::copy_on_write_t<arrays::array_t> > map_type;
    typedef std::pair<core::name_t, core::copy_on_write_t<arrays::array_t> >                  pair_type;

    typedef map_type::const_iterator    const_iterator;
    typedef map_type::iterator          iterator;

    map_type map;
};

attribute_table_t::attribute_table_t()
{
    init();
    pimpl_ = new impl();
}

attribute_table_t::~attribute_table_t() { delete pimpl_;}

attribute_table_t::attribute_table_t( const attribute_table_t& other)
{
    assert( other.pimpl_);

    pimpl_ = new impl( *other.pimpl_);
}

void attribute_table_t::init() { pimpl_ = 0;}

void attribute_table_t::swap( attribute_table_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

attribute_table_t attribute_table_t::create_empty_with_same_attributes( const attribute_table_t& other)
{
    assert( other.pimpl_);

    attribute_table_t new_table;
    for( impl::const_iterator it( other.pimpl_->map.begin()), e( other.pimpl_->map.end()); it != e; ++it)
        new_table.insert( it->first, it->second.read().type());

    return new_table;
}

bool attribute_table_t::empty() const
{
    assert( pimpl_);

    return pimpl_->map.empty();
}

std::size_t attribute_table_t::size() const
{
    assert( pimpl_);

    return pimpl_->map.size();
}

void attribute_table_t::clear()
{
    assert( pimpl_);

    pimpl_->map.clear();
}

bool attribute_table_t::has_attribute( const core::name_t& name) const
{
    assert( pimpl_);

    if( pimpl_->map.find( name) != pimpl_->map.end())
        return true;

    return false;
}

core::type_t attribute_table_t::attribute_type( const core::name_t& name) const
{
    assert( pimpl_);

    impl::const_iterator it = pimpl_->map.find( name);

    if( it != pimpl_->map.end())
        return it->second.read().type();

    throw attribute_not_found( name);

    // return anything, to keep dumb compilers happy
    return core::float_k;
}

void attribute_table_t::insert( const core::name_t& name, core::type_t attr_type)
{
    arrays::array_t array( attr_type);
    core::copy_on_write_t<arrays::array_t> cow_array( boost::move( array));
    pimpl_->map[name] = cow_array;
}

void attribute_table_t::insert( const core::name_t& name, const arrays::array_t& array)
{
    core::copy_on_write_t<arrays::array_t> cow_array( array);
    pimpl_->map[name] = cow_array;
}

void attribute_table_t::insert( const core::name_t& name, BOOST_RV_REF( arrays::array_t) array)
{
    core::copy_on_write_t<arrays::array_t> cow_array( array);
    pimpl_->map[name] = cow_array;
}

void attribute_table_t::erase( const core::name_t& name)
{
    assert( pimpl_);

    pimpl_->map.erase( name);
}

void attribute_table_t::erase_arrays_indices( unsigned int start, unsigned int end)
{
    for( impl::iterator it( pimpl_->map.begin()), e( pimpl_->map.end()); it != e; ++it)
        it->second.write().erase( start, end);
}

void attribute_table_t::erase_arrays_index( unsigned int index)
{
    erase_arrays_indices( index, index + 1);
}

const_attribute_ref_t attribute_table_t::get_const_attribute_ref( const core::name_t& name) const
{
    assert( pimpl_);

    impl::const_iterator it = pimpl_->map.find( name);

    if( it != pimpl_->map.end())
        return const_attribute_ref_t( &( it->second.read()));

    return const_attribute_ref_t( 0);
}

attribute_ref_t attribute_table_t::get_attribute_ref( const core::name_t& name)
{
    assert( pimpl_);

    impl::iterator it = pimpl_->map.find( name);

    if( it != pimpl_->map.end())
        return attribute_ref_t( &( it->second.write()));

    return attribute_ref_t( 0);
}

const arrays::array_t& attribute_table_t::const_array( const core::name_t& name) const
{
    const_attribute_ref_t ref( get_const_attribute_ref( name) );

    if( !ref.valid())
        throw attribute_not_found( name);

    return ref.array();
}

arrays::array_t& attribute_table_t::array( const core::name_t& name)
{
    attribute_ref_t ref( get_attribute_ref( name) );

    if( !ref.valid())
        throw attribute_not_found( name);

    return ref.array();
}

// iterators

attribute_table_t::const_iterator::const_iterator() : pimpl_( 0) {}

attribute_table_t::const_iterator::const_iterator( void *pimpl) : pimpl_( pimpl) {}

attribute_table_t::const_iterator& attribute_table_t::const_iterator::operator++()
{
    typedef attribute_table_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) + 1);
    return *this;
}

attribute_table_t::const_iterator attribute_table_t::const_iterator::operator++( int)
{
    const_iterator saved( *this);
    ++( *this);
    return saved;
}

attribute_table_t::const_iterator& attribute_table_t::const_iterator::operator--()
{
    typedef attribute_table_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) - 1);
    return *this;
}

attribute_table_t::const_iterator attribute_table_t::const_iterator::operator--( int)
{
    const_iterator saved( *this);
    --( *this);
    return saved;
}

const core::name_t& attribute_table_t::const_iterator::first() const
{
    assert( pimpl_);

    typedef attribute_table_t::impl::pair_type pair_type;
    return reinterpret_cast<const pair_type*>( pimpl_)->first;
}

const arrays::array_t& attribute_table_t::const_iterator::second() const
{
    assert( pimpl_);

    typedef attribute_table_t::impl::pair_type pair_type;
    return reinterpret_cast<const pair_type*>( pimpl_)->second.read();
}

bool attribute_table_t::const_iterator::operator==( const const_iterator& other) const
{
    return pimpl_ == other.pimpl_;
}

bool attribute_table_t::const_iterator::operator!=( const const_iterator& other) const
{
    return !( *this == other);
}

attribute_table_t::const_iterator attribute_table_t::begin() const
{
    assert( pimpl_);

    return const_iterator( pimpl_->map.begin().get_ptr());
}

attribute_table_t::const_iterator attribute_table_t::end() const
{
    assert( pimpl_);

    return const_iterator( pimpl_->map.end().get_ptr());
}

attribute_table_t::iterator::iterator() : pimpl_( 0) {}

attribute_table_t::iterator::iterator( void *pimpl) : pimpl_( pimpl) {}

attribute_table_t::iterator& attribute_table_t::iterator::operator++()
{
    typedef attribute_table_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) + 1);
    return *this;
}

attribute_table_t::iterator attribute_table_t::iterator::operator++( int)
{
    iterator saved( *this);
    ++( *this);
    return saved;
}

attribute_table_t::iterator& attribute_table_t::iterator::operator--()
{
    typedef attribute_table_t::impl::pair_type pair_type;

    pimpl_ = reinterpret_cast<void*>( reinterpret_cast<pair_type*>( pimpl_) - 1);
    return *this;
}

attribute_table_t::iterator attribute_table_t::iterator::operator--( int)
{
    iterator saved( *this);
    --( *this);
    return saved;
}

const core::name_t& attribute_table_t::iterator::first() const
{
    assert( pimpl_);

    typedef attribute_table_t::impl::pair_type pair_type;
    return reinterpret_cast<const pair_type*>( pimpl_)->first;
}

arrays::array_t& attribute_table_t::iterator::second()
{
    assert( pimpl_);

    typedef attribute_table_t::impl::pair_type pair_type;
    return reinterpret_cast<pair_type*>( pimpl_)->second.write();
}

bool attribute_table_t::iterator::operator==( const iterator& other) const
{
    return pimpl_ == other.pimpl_;
}

bool attribute_table_t::iterator::operator!=( const iterator& other) const
{
    return !( *this == other);
}

attribute_table_t::iterator attribute_table_t::begin()
{
    assert( pimpl_);

    return iterator( pimpl_->map.begin().get_ptr());
}

attribute_table_t::iterator attribute_table_t::end()
{
    assert( pimpl_);

    return iterator( pimpl_->map.end().get_ptr());
}

namespace
{

class push_back_element_copy_visitor
{
public:

    push_back_element_copy_visitor( const arrays::array_t& src_array,
                                    std::size_t element_index) : src_array_( src_array)
    {
        element_index_ = element_index;
    }

    template<class T>
    void operator()( arrays::array_ref_t<T>& array)
    {
        //arrays::const_array_ref_t<T> src_ref( src_array_);
        //array.push_back( src_ref[element_index_]);
    }

private:

    const arrays::array_t& src_array_;
    std::size_t element_index_;
};

} // unnamed

void attribute_table_t::push_back_attribute_values_copy( const attribute_table_t& src,
                                                         std::size_t element_index)
{
    for( impl::iterator it( pimpl_->map.begin()), e( pimpl_->map.end()); it != e; ++it)
    {
        const arrays::array_t& src_array( src.const_array( it->first));
        assert( element_index < src_array.size());

        push_back_element_copy_visitor v( src_array, element_index);
        arrays::apply_visitor( v, it->second.write());
    }
}

bool attribute_table_t::operator==( const attribute_table_t& other) const
{
    assert( pimpl_);
    assert( other.pimpl_);

    return pimpl_->map == other.pimpl_->map;
}

bool attribute_table_t::operator!=( const attribute_table_t& other) const
{
    return !( *this == other);
}

bool attribute_table_t::check_consistency() const
{
    assert( pimpl_);

    // Check that all arrays have the same length.
    int array_size = -1;

    for( impl::const_iterator it( pimpl_->map.begin()), e( pimpl_->map.end()); it != e; ++it)
    {
        if( array_size > 0 && it->second.read().size() != array_size)
            return false;
        else
            array_size = it->second.read().size();
    }

    return true;
}

} // geo
} // ramen
