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

#include<ramen/geo/visitors/count_ngons.hpp>

#include<ramen/geo/shape.hpp>
#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>

namespace ramen
{
namespace geo
{

count_ngons_visitor::count_ngons_visitor() : const_shape_visitor()
{
}

void count_ngons_visitor::visit( const poly_mesh_model_t& model, const shape_t& shape)
{
    do_visit( model);
}

void count_ngons_visitor::visit( const subd_mesh_model_t& model, const shape_t& shape)
{
    do_visit( model);
}

void count_ngons_visitor::do_visit( const mesh_model_t& model)
{
    num_degenerate = 0;
    num_triangles = 0;
    num_quads = 0;
    num_ngons = 0;

    arrays::const_array_ref_t<boost::uint32_t> verts_per_face( model.const_verts_per_face_array());

    for( int i = 0, e = verts_per_face.size(); i < e; ++i)
    {
        switch( verts_per_face[i])
        {
            case 0:
            case 1:
            case 2:
                num_degenerate++;
            break;

            case 3:
                num_triangles++;
            break;

            case 4:
                num_quads++;
            break;

            default:
                num_ngons++;
            break;
        }
    }
}

} // geo
} // ramen
