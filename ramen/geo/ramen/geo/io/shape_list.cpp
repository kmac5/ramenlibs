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

#include<ramen/geo/io/shape_list.hpp>

#include<algorithm>

#include<boost/container/vector.hpp>

#include<ramen/core/memory.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

shape_entry_t::shape_entry_t( const core::string8_t& shape_name,
                              boost::uint32_t shape_type,
                              const math::box3d_t& shape_bbox) : name( shape_name),
                                                                 type( shape_type),
                                                                 bbox( shape_bbox)
{
}

const char *shape_entry_t::type_string() const
{
    return shape_type_to_string( (shape_type_t) type);
}

struct shape_list_filter_t::impl
{
    impl() : regex_( ".*"), type_mask_( ~0)
    {
    }

    impl( const core::regex8_t& regex) : regex_( regex), type_mask_( ~0)
    {
    }

    impl( boost::uint32_t type_mask) : regex_( ".*"), type_mask_( type_mask)
    {
    }

    impl( const core::regex8_t& regex, boost::uint32_t type_mask) : regex_( regex), type_mask_( type_mask)
    {
    }

    boost::uint32_t type_mask_;
    core::regex8_t regex_;
};

shape_list_filter_t::shape_list_filter_t() : pimpl_( 0) {}

shape_list_filter_t::shape_list_filter_t( const core::regex8_t& regex)
{
    pimpl_ = new impl( regex);
}

shape_list_filter_t::shape_list_filter_t( boost::uint32_t type_mask)
{
    pimpl_ = new impl( type_mask);
}

shape_list_filter_t::shape_list_filter_t( const core::regex8_t& regex, boost::uint32_t type_mask)
{
    pimpl_ = new impl( regex, type_mask);
}

shape_list_filter_t::~shape_list_filter_t() { delete pimpl_;}

boost::uint32_t shape_list_filter_t::type_mask() const
{
    if( pimpl_)
        return pimpl_->type_mask_;

    return ~0;
}

bool shape_list_filter_t::operator()( const shape_entry_t& entry) const
{
    return (*this)( entry.name, entry.type);
}

bool shape_list_filter_t::operator()( const core::string8_t& name, boost::uint32_t type) const
{
    if( !( type & type_mask()))
        return false;

    if( !regex_match( name))
        return false;

    return true;
}

bool shape_list_filter_t::regex_match( const core::string8_t& name) const
{
    if( pimpl_)
        return pimpl_->regex_.full_match( name);

    return true;
}

struct shape_list_t::impl
{
    boost::container::vector<shape_entry_t> vector_;
};

shape_list_t::shape_list_t()
{
    init();
    pimpl_ = new impl();
}

shape_list_t::~shape_list_t() { delete pimpl_;}

shape_list_t::shape_list_t( const shape_list_t& other)
{
    assert( other.pimpl_);

    pimpl_ = new impl( *other.pimpl_);
}

void shape_list_t::init()
{
    pimpl_ = 0;
}

void shape_list_t::swap( shape_list_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

void shape_list_t::emplace_back( const core::string8_t& shape_name, boost::uint32_t shape_type)
{
    assert( pimpl_);

    pimpl_->vector_.emplace_back( shape_name, shape_type);
}

void shape_list_t::push_back( const shape_entry_t& entry)
{
    assert( pimpl_);

    pimpl_->vector_.push_back( entry);
}

shape_list_t::const_iterator shape_list_t::begin() const
{
    assert( pimpl_);

    return pimpl_->vector_.begin().get_ptr();
}

shape_list_t::const_iterator shape_list_t::end() const
{
    assert( pimpl_);

    return pimpl_->vector_.end().get_ptr();
}

} // io
} // geo
} // ramen
