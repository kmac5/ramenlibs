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

#ifndef RAMEN_GEO_VISITORS_TRIANGULATE_HPP
#define RAMEN_GEO_VISITORS_TRIANGULATE_HPP

#include<ramen/geo/shape_models/shape_visitor.hpp>

#include<ramen/geo/shape.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API triangulate_visitor : public shape_visitor
{
public:

    explicit triangulate_visitor( bool keep_quads = false);

    virtual void visit( poly_mesh_model_t& model, shape_t& shape);
    virtual void visit( subd_mesh_model_t& model, shape_t& shape);

private:

    void do_visit( mesh_model_t& model,
                   shape_t& shape,
                   mesh_model_t& new_model,
                   shape_t& new_shape);

    void copy_face( mesh_model_t& model,
                    const shape_t& shape,
                    std::size_t index,
                    boost::uint32_t face_start_index,
                    mesh_model_t& new_model,
                    shape_t& new_shape);

    void triangulate_quad( mesh_model_t& model,
                           const shape_t& shape,
                           std::size_t index,
                           boost::uint32_t face_start_index,
                           mesh_model_t& new_model,
                           shape_t& new_shape);

    bool keep_quads_;
};

} // geo
} // ramen

#endif
