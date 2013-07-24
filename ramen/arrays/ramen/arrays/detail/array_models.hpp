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

#ifndef RAMEN_ARRAYS_DETAIL_ARRAY_MODELS_HPP
#define RAMEN_ARRAYS_DETAIL_ARRAY_MODELS_HPP

#include<ramen/arrays/detail/array_interface.hpp>

#include<algorithm>

#include<ramen/core/Concepts/RegularConcept.hpp>
#include<ramen/arrays/Concepts/ArrayModelConcept.hpp>

namespace ramen
{
namespace arrays
{
namespace detail
{

template<class T> // T is boost::container::vector<>
class array_model_t : public array_interface_t
{
    BOOST_CONCEPT_ASSERT(( ArrayModelConcept<T>));
    BOOST_CONCEPT_ASSERT(( core::RegularConcept<typename T::value_type>));

public:

    typedef typename T::value_type value_type;

    array_model_t() {}

    virtual array_model_t<T> *copy() const
    {
        return new array_model_t( *this);
    }

    virtual core::type_t type() const
    {
        return core::type_traits<value_type>::type();
    }

    virtual bool empty() const  { return items_.empty();}
    virtual std::size_t size() const { return items_.size();}
    virtual std::size_t capacity() const { return items_.capacity();}

    virtual void clear() { items_.clear();}

    virtual void reserve( std::size_t n) { items_.reserve( n);}

    virtual void erase( std::size_t start, std::size_t end)
    {
        items_.erase( items_.begin() + start, items_.begin() + end);
    }

    virtual void shrink_to_fit() { items_.shrink_to_fit();}

    virtual std::size_t item_size() const
    {
        return sizeof( value_type);
    }

    virtual void copy_element( std::size_t from, std::size_t to)
    {
        items_[to] = items_[from];
    }

    virtual void push_back( const void *x)
    {
        items_.push_back( *( reinterpret_cast<const value_type*>( x)));
    }

    virtual void resize( std::size_t n)
    {
        items_.resize( n, core::type_traits<value_type>::default_value());
    }

    virtual bool equal( const array_interface_t& other) const
    {
        const array_model_t<T> *other_model = static_cast<const array_model_t<T>*>( &other);
        return items_ == other_model->items_;
    }

    T items_;
};

} // detail
} // arrays
} // ramen

#endif
