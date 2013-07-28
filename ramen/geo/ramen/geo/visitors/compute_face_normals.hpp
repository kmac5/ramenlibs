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

#ifndef RAMEN_GEO_VISITORS_FACE_NORMALS_HPP
#define RAMEN_GEO_VISITORS_FACE_NORMALS_HPP

#include<ramen/geo/shape_models/shape_visitor.hpp>

#include<ramen/geo/shape.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API compute_face_normals_visitor : public shape_visitor
{
public:

    compute_face_normals_visitor();

    virtual void visit( const poly_mesh_model_t& model, shape_t& shape);
    virtual void visit( const visitable_t& model, shape_t& shape);
};

} // geo
} // ramen

#endif
