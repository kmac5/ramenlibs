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

#include<ramen/arrays/array.hpp>

#include<boost/container/vector.hpp>

#include<ramen/core/exceptions.hpp>

#include<ramen/arrays/detail/array_models.hpp>
#include<ramen/arrays/detail/string_array_model.hpp>

namespace ramen
{
namespace arrays
{

array_t::array_t() : model_( 0)
{
    init( core::float_k, 0);
}

array_t::array_t( core::type_t type) : model_( 0)
{
    init( type, 0);
}

array_t::array_t( core::type_t type, size_type n) : model_( 0)
{
    init( type, n);
}

void array_t::init( core::type_t type, size_type n)
{
    #define RAMEN_ARRAYS_ARRAY_INIT_CASE( type_enum_k, type_name)\
        case type_enum_k:\
            do_init<type_name>();\
        break;

    switch( type)
    {
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::bool_k, bool)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::float_k, float)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::double_k, double)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::int8_k, boost::int8_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::uint8_k, boost::uint8_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::int16_k, boost::int16_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::uint16_k, boost::uint16_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::int32_k, boost::int32_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::uint32_k, boost::uint32_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::int64_k, boost::int64_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::uint64_k, boost::uint64_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::point2f_k, math::point2f_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::point3f_k, math::point3f_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::vector2i_k, math::vector2i_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::vector2f_k, math::vector2f_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::vector3f_k, math::vector3f_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::normalf_k, math::normalf_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::hpoint2f_k, math::hpoint2f_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::hpoint3f_k, math::hpoint3f_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::box2i_k, math::box2i_t)
        RAMEN_ARRAYS_ARRAY_INIT_CASE( core::color3f_k, color::color3f_t)

        #if RAMEN_WITH_HALF
            RAMEN_ARRAYS_ARRAY_INIT_CASE( core::half_k, half)
            RAMEN_ARRAYS_ARRAY_INIT_CASE( core::point2h_k, math::point2h_t)
            RAMEN_ARRAYS_ARRAY_INIT_CASE( core::point3h_k, math::point3h_t)
            RAMEN_ARRAYS_ARRAY_INIT_CASE( core::vector2h_k, math::vector2h_t)
            RAMEN_ARRAYS_ARRAY_INIT_CASE( core::vector3h_k, math::vector3h_t)
            RAMEN_ARRAYS_ARRAY_INIT_CASE( core::normalh_k, math::normalh_t)
        #endif

        // special case for strings
        case core::string8_k:
            do_init_with_string();
        break;

        default:
            throw core::not_implemented();
    }

    if( n != 0)
        resize( n);

    #undef RAMEN_ARRAYS_ARRAY_INIT_CASE
}

template<class T>
void array_t::do_init()
{
    model_ = new detail::array_model_t<boost::container::vector<T> >();
}

void array_t::do_init_with_string()
{
    model_ = new detail::string_array_model_t();
}

array_t::~array_t() { delete model_;}

array_t::array_t( const array_t& other) : model_( 0)
{
    assert( other.model_);

    model_ = other.model_->copy();
}

void array_t::swap( array_t& other)
{
    std::swap( model_, other.model_);
}

core::type_t array_t::type() const
{
    assert( model_);

    return model_->type();
}

bool array_t::empty() const
{
    assert( model_);

    return model_->empty();
}

array_t::size_type array_t::size() const
{
    assert( model_);

    return model_->size();
}

array_t::size_type array_t::capacity() const
{
    assert( model_);

    return model_->capacity();
}

void array_t::clear()
{
    assert( model_);

    return model_->clear();
}

void array_t::reserve( size_type n)
{
    assert( model_);

    model_->reserve( n);
}

void array_t::erase( size_type start, size_type end)
{
    assert( model_);
    assert( end >= start);
    assert( end <= size());

    // TODO: check for exceptions here.
    model_->erase( start, end);
}

void array_t::erase( size_type pos)
{
    erase( pos, pos + 1);
}

void array_t::shrink_to_fit()
{
    assert( model_);

    model_->shrink_to_fit();
}

void array_t::copy_element( size_type from, size_type to)
{
    assert( model_);
    assert( from < size());
    assert( to < size());

    model_->copy_element( from, to);
}

const detail::array_interface_t *array_t::model() const
{
    return model_;
}

detail::array_interface_t *array_t::model()
{
    return model_;
}

void array_t::push_back( const void *x)
{
    assert( model_);

    model_->push_back( x);
}

void array_t::resize( size_type n)
{
    assert( model_);

    model_->resize( n);
}

bool array_t::operator==( const array_t& other) const
{
    if( model_ == 0 || other.model_ == 0)
    {
        return model_ == other.model_;
    }

    if( type() != other.type())
        return false;

    return model_->equal( *other.model_);
}

bool array_t::operator!=( const array_t& other) const
{
    return !( *this == other);
}

} // arrays
} // ramen
