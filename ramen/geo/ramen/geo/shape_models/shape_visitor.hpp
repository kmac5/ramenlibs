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

#ifndef RAMEN_GEO_SHAPE_MODEL_VISITOR_HPP
#define RAMEN_GEO_SHAPE_MODEL_VISITOR_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_fwd.hpp>

#include<ramen/geo/shape_models/visitable.hpp>
#include<ramen/geo/shape_models/mesh_model_fwd.hpp>
#include<ramen/geo/shape_models/poly_mesh_model_fwd.hpp>
#include<ramen/geo/shape_models/subd_mesh_model_fwd.hpp>
#include<ramen/geo/shape_models/nurbs_curve_model_fwd.hpp>
#include<ramen/geo/shape_models/nurbs_surface_model_fwd.hpp>
#include<ramen/geo/shape_models/curves_model_fwd.hpp>
#include<ramen/geo/shape_models/points_model_fwd.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API const_shape_visitor
{
public:

    explicit const_shape_visitor( bool nothrow = false);
    virtual ~const_shape_visitor();

    virtual void visit( const poly_mesh_model_t& model, const shape_t& shape);
    virtual void visit( const subd_mesh_model_t& model, const shape_t& shape);
    virtual void visit( const nurbs_curve_model_t& model, const shape_t& shape);
    virtual void visit( const nurbs_surface_model_t& model, const shape_t& shape);
    virtual void visit( const curves_model_t& model, const shape_t& shape);
    virtual void visit( const points_model_t& model, const shape_t& shape);

    // catch all overload.
    virtual void visit( const visitable_t& model, const shape_t& shape);

protected:

    bool nothrow_;
};

class RAMEN_GEO_API shape_visitor
{
public:

    explicit shape_visitor( bool nothrow = false);
    virtual ~shape_visitor();

    virtual void visit( poly_mesh_model_t& model, shape_t& shape);
    virtual void visit( subd_mesh_model_t& model, shape_t& shape);
    virtual void visit( nurbs_curve_model_t& model, shape_t& shape);
    virtual void visit( nurbs_surface_model_t& model, shape_t& shape);
    virtual void visit( curves_model_t& model, shape_t& shape);
    virtual void visit( points_model_t& model, shape_t& shape);

    // catch all overload.
    virtual void visit( visitable_t& model, shape_t& shape);

protected:

    bool nothrow_;
};

} // geo
} // ramen

#endif
