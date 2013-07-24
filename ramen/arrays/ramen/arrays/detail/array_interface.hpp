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

#ifndef RAMEN_ARRAYS_DETAIL_ARRAY_INTERFACE_HPP
#define RAMEN_ARRAYS_DETAIL_ARRAY_INTERFACE_HPP

#include<ramen/arrays/detail/array_interface_fwd.hpp>

#include<cstddef>

#include<ramen/core/types.hpp>

namespace ramen
{
namespace arrays
{
namespace detail
{

struct array_interface_t
{
    virtual ~array_interface_t() {}

    virtual array_interface_t *copy() const = 0;

    virtual core::type_t type() const = 0;

    virtual bool empty() const = 0;
    virtual std::size_t size() const = 0;
    virtual std::size_t capacity() const = 0;

    virtual void clear() = 0;
    virtual void reserve( std::size_t n) = 0;
    virtual void erase( std::size_t start, std::size_t end) = 0;
    virtual void shrink_to_fit() = 0;
    virtual void copy_element( std::size_t from, std::size_t to) = 0;

    virtual std::size_t item_size() const = 0;

    virtual void push_back( const void *x) = 0;
    virtual void resize( std::size_t n) = 0;

    virtual bool equal( const array_interface_t& other) const = 0;
};

} // detail
} // arrays
} // ramen

#endif
