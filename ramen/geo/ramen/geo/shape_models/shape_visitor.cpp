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

#include<ramen/geo/shape_models/shape_visitor.hpp>

#include<ramen/geo/shape.hpp>
#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>
#include<ramen/geo/shape_models/nurbs_curve_model.hpp>
#include<ramen/geo/shape_models/nurbs_surface_model.hpp>
#include<ramen/geo/shape_models/curves_model.hpp>
#include<ramen/geo/shape_models/points_model.hpp>

namespace ramen
{
namespace geo
{

const_shape_visitor::const_shape_visitor() {}
const_shape_visitor::~const_shape_visitor() {}

void const_shape_visitor::visit( const poly_mesh_model_t& model, const shape_t& shape)
{
    visit( static_cast<const visitable_t&>( model), shape);
}

void const_shape_visitor::visit( const subd_mesh_model_t& model, const shape_t& shape)
{
    visit( static_cast<const visitable_t&>( model), shape);
}

void const_shape_visitor::visit( const nurbs_curve_model_t& model, const shape_t& shape)
{
    visit( static_cast<const visitable_t&>( model), shape);
}

void const_shape_visitor::visit( const nurbs_surface_model_t& model, const shape_t& shape)
{
    visit( static_cast<const visitable_t&>( model), shape);
}

void const_shape_visitor::visit( const curves_model_t& model, const shape_t& shape)
{
    visit( static_cast<const visitable_t&>( model), shape);
}

void const_shape_visitor::visit( const points_model_t& model, const shape_t& shape)
{
    visit( static_cast<const visitable_t&>( model), shape);
}

void const_shape_visitor::visit( const visitable_t& model, const shape_t& shape)
{
    throw bad_shape_type( shape.type());
}

shape_visitor::shape_visitor() {}
shape_visitor::~shape_visitor() {}

void shape_visitor::visit( poly_mesh_model_t& model, shape_t& shape)
{
    visit( static_cast<visitable_t&>( model), shape);
}

void shape_visitor::visit( subd_mesh_model_t& model, shape_t& shape)
{
    visit( static_cast<visitable_t&>( model), shape);
}

void shape_visitor::visit( nurbs_curve_model_t& model, shape_t& shape)
{
    visit( static_cast<visitable_t&>( model), shape);
}

void shape_visitor::visit( nurbs_surface_model_t& model, shape_t& shape)
{
    visit( static_cast<visitable_t&>( model), shape);
}

void shape_visitor::visit( curves_model_t& model, shape_t& shape)
{
    visit( static_cast<visitable_t&>( model), shape);
}

void shape_visitor::visit( points_model_t& model, shape_t& shape)
{
    visit( static_cast<visitable_t&>( model), shape);
}

void shape_visitor::visit( visitable_t& model, shape_t& shape)
{
    throw bad_shape_type( shape.type());
}

} // geo
} // ramen
