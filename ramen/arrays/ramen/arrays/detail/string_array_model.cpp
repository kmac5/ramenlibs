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

#include<ramen/arrays/detail/string_array_model.hpp>

namespace ramen
{
namespace arrays
{
namespace detail
{

string_array_model_t::string_array_model_t() {}

string_array_model_t *string_array_model_t::copy() const
{
    return new string_array_model_t( *this);
}

core::type_t string_array_model_t::type() const
{
    return core::type_traits<value_type>::type();
}

bool string_array_model_t::empty() const  { return items_.empty();}
std::size_t string_array_model_t::size() const { return items_.size();}
std::size_t string_array_model_t::capacity() const { return items_.capacity();}

void string_array_model_t::clear() { items_.clear();}

void string_array_model_t::reserve( std::size_t n) { items_.reserve( n);}

void string_array_model_t::erase( std::size_t start, std::size_t end)
{
    items_.erase( start, end);
}

void string_array_model_t::shrink_to_fit() { items_.shrink_to_fit();}

std::size_t string_array_model_t::item_size() const
{
    return sizeof( value_type);
}

void string_array_model_t::copy_element( std::size_t from, std::size_t to)
{
    items_[to] = items_[from];
}

void string_array_model_t::push_back( const void *x)
{
    items_.push_back( *( reinterpret_cast<const value_type*>( x)));
}

void string_array_model_t::resize( std::size_t n)
{
    items_.resize( n);
}

bool string_array_model_t::equal( const array_interface_t& other) const
{
    const string_array_model_t *other_model = static_cast<const string_array_model_t*>( &other);
    return items_ == other_model->items_;
}

} // detail
} // arrays
} // ramen
