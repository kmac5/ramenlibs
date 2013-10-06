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

#ifndef RAMEN_ARRAYS_DETAIL_STRING_ARRAY_MODEL_HPP
#define RAMEN_ARRAYS_DETAIL_STRING_ARRAY_MODEL_HPP

#include<ramen/arrays/detail/array_interface.hpp>

#include<algorithm>

#include<ramen/core/Concepts/RegularConcept.hpp>
#include<ramen/core/allocator_interface.hpp>
#include<ramen/containers/string8_vector.hpp>

#include<ramen/arrays/Concepts/ArrayModelConcept.hpp>

namespace ramen
{
namespace arrays
{
namespace detail
{

class string_array_model_t : public array_interface_t
{
public:

    typedef core::string8_t value_type;

    explicit string_array_model_t( const core::allocator_ptr_t& alloc);

    virtual string_array_model_t *copy() const;

    virtual core::type_t type() const;

    virtual bool empty() const;
    virtual std::size_t size() const;
    virtual std::size_t capacity() const;

    virtual void clear();

    virtual void reserve( std::size_t n);

    virtual void erase( std::size_t start, std::size_t end);
    virtual void shrink_to_fit();

    virtual std::size_t item_size() const;

    virtual void copy_element( std::size_t from, std::size_t to);

    virtual void push_back( const void *x);

    virtual void resize( std::size_t n);

    virtual bool equal( const array_interface_t& other) const;

    containers::string8_vector_t items_;
};

} // detail
} // arrays
} // ramen

#endif
