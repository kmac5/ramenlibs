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

#include<ramen/geo/shape_attributes.hpp>

namespace ramen
{
namespace geo
{

shape_attributes_t::shape_attributes_t() {}

void shape_attributes_t::swap( shape_attributes_t& other)
{
    point_attrs_.swap( other.point_attrs_);
    vertex_attrs_.swap( other.vertex_attrs_);
    primitive_attrs_.swap( other.primitive_attrs_);
    constant_attrs_.swap( other.constant_attrs_);
}

const attribute_table_t& shape_attributes_t::point() const
{
    return point_attrs_;
}

attribute_table_t& shape_attributes_t::point()
{
    return point_attrs_;
}

const attribute_table_t& shape_attributes_t::vertex() const
{
    return vertex_attrs_;
}

attribute_table_t& shape_attributes_t::vertex()
{
    return vertex_attrs_;
}

const attribute_table_t& shape_attributes_t::primitive() const
{
    return primitive_attrs_;
}

attribute_table_t& shape_attributes_t::primitive()
{
    return primitive_attrs_;
}

const core::dictionary_t& shape_attributes_t::constant() const
{
    return constant_attrs_;
}

core::dictionary_t& shape_attributes_t::constant()
{
    return constant_attrs_;
}

bool shape_attributes_t::operator==( const shape_attributes_t& other) const
{
    return  point() == other.point() &&
            vertex() == other.vertex() &&
            primitive() == other.primitive() &&
            constant() == other.constant();
}

bool shape_attributes_t::operator!=( const shape_attributes_t& other) const
{
    return !( *this == other);
}

bool shape_attributes_t::check_consistency() const
{
    return  point().check_consistency() &&
            vertex().check_consistency() &&
            primitive().check_consistency();
}

} // geo
} // ramen
