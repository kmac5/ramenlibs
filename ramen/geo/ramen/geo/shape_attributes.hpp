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

#ifndef RAMEN_GEO_SHAPE_ATTRIBUTES_HPP
#define RAMEN_GEO_SHAPE_ATTRIBUTES_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_attributes_fwd.hpp>

#include<ramen/core/dictionary.hpp>

#include<ramen/geo/attribute_table.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API shape_attributes_t
{
public:

    shape_attributes_t();

    void swap( shape_attributes_t& other);

    const attribute_table_t& point() const;
    attribute_table_t& point();

    const attribute_table_t& vertex() const;
    attribute_table_t& vertex();

    const attribute_table_t& primitive() const;
    attribute_table_t& primitive();

    const core::dictionary_t& constant() const;
    core::dictionary_t& constant();

    bool operator==( const shape_attributes_t& other) const;
    bool operator!=( const shape_attributes_t& other) const;

    bool check_consistency() const;

private:

    attribute_table_t point_attrs_;
    attribute_table_t vertex_attrs_;
    attribute_table_t primitive_attrs_;
    core::dictionary_t constant_attrs_;
};

} // geo
} // ramen

#endif
