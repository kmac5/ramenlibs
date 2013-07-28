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

#include<ramen/geo/shape_vector.hpp>

#include<algorithm>

#include<boost/container/vector.hpp>
#include<boost/bind.hpp>
#include<boost/foreach.hpp>

#include<ramen/core/memory.hpp>

namespace ramen
{
namespace geo
{

struct shape_vector_t::impl
{
    boost::container::vector<shape_t> vector_;
};


shape_vector_t::shape_vector_t()
{
    init();
    pimpl_ = new impl();
}

shape_vector_t::~shape_vector_t() { delete pimpl_;}

shape_vector_t::shape_vector_t( const shape_vector_t& other)
{
    assert( other.pimpl_);

    pimpl_ = new impl( *other.pimpl_);
}

void shape_vector_t::init()
{
    pimpl_ = 0;
}

void shape_vector_t::swap( shape_vector_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

bool shape_vector_t::empty() const
{
    assert( pimpl_);

    return pimpl_->vector_.empty();
}

std::size_t shape_vector_t::size() const
{
    assert( pimpl_);

    return pimpl_->vector_.size();
}

void shape_vector_t::clear()
{
    assert( pimpl_);

    pimpl_->vector_.clear();
}

const shape_t& shape_vector_t::operator[]( unsigned int index) const
{
    assert( pimpl_);
    assert( index < size());

    return pimpl_->vector_[index];
}

shape_t& shape_vector_t::operator[]( unsigned int index)
{
    assert( pimpl_);
    assert( index < size());

    return pimpl_->vector_[index];
}

const shape_t& shape_vector_t::back() const
{
    assert( pimpl_);

    return pimpl_->vector_.back();
}

shape_t& shape_vector_t::back()
{
    assert( pimpl_);

    return pimpl_->vector_.back();
}

void shape_vector_t::push_back( const shape_t& shape)
{
    assert( pimpl_);

    pimpl_->vector_.push_back( shape);
}

void shape_vector_t::push_back( BOOST_RV_REF( shape_t) shape)
{
    assert( pimpl_);

    pimpl_->vector_.push_back( shape);
}

shape_vector_t::const_iterator shape_vector_t::begin() const
{
    assert( pimpl_);

    return pimpl_->vector_.begin().get_ptr();
}

shape_vector_t::const_iterator shape_vector_t::end() const
{
    assert( pimpl_);

    return pimpl_->vector_.end().get_ptr();
}

shape_vector_t::iterator shape_vector_t::begin()
{
    assert( pimpl_);

    return pimpl_->vector_.begin().get_ptr();
}

shape_vector_t::iterator shape_vector_t::end()
{
    assert( pimpl_);

    return pimpl_->vector_.end().get_ptr();
}

void shape_vector_t::remove_invalid_shapes()
{
    assert( pimpl_);

    pimpl_->vector_.erase( std::remove_if(  pimpl_->vector_.begin(),
                                            pimpl_->vector_.end(),
                                            !boost::bind( &shape_t::check_consistency, _1)),
                                            pimpl_->vector_.end());
}

void apply_visitor( const_shape_visitor& v, const shape_vector_t& obj_list)
{
    BOOST_FOREACH( const shape_t& s, obj_list)
        apply_visitor( v, s);
}

void apply_visitor( shape_visitor& v, shape_vector_t& obj_list)
{
    BOOST_FOREACH( shape_t& s, obj_list)
        apply_visitor( v, s);
}

} // geo
} // ramen
