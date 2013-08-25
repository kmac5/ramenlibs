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

#include<ramen/geo/visitors/compute_face_normals.hpp>

#include<boost/iterator/permutation_iterator.hpp>

#include<ramen/geo/shape.hpp>
#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>
#include<ramen/geo/global_names.hpp>
#include<ramen/geo/algorithm/polygon_normal.hpp>

namespace ramen
{
namespace geo
{

compute_face_normals_visitor::compute_face_normals_visitor() : shape_visitor()
{
}

void compute_face_normals_visitor::visit( poly_mesh_model_t& model, shape_t& shape)
{
    do_visit( model, shape);
}

void compute_face_normals_visitor::visit( subd_mesh_model_t& model, shape_t& shape)
{
    do_visit( model, shape);
}

void compute_face_normals_visitor::visit( visitable_t& model, shape_t& shape)
{
    // ignore all other shapes.
}

void compute_face_normals_visitor::do_visit( mesh_model_t& model, shape_t& shape)
{
    if( !shape.attributes().point().has_attribute( g_P_name))
        throw core::runtime_error( core::string8_t( "No P attribute found in compute face normals visitor"));

    // create the N array, if it does not exist.
    if( shape.attributes().primitive().has_attribute( g_N_name))
    {
        arrays::array_t N( core::normalf_k);
        N.reserve( model.num_faces());
        shape.attributes().primitive().insert( g_N_name, N);
    }

    arrays::const_array_ref_t<math::point3f_t> points( shape.attributes().point().const_array( g_P_name));
    arrays::const_array_ref_t<boost::uint32_t> verts_per_face( model.const_verts_per_face_array());
    arrays::const_array_ref_t<boost::uint32_t> face_vert_indices( model.const_face_vert_indices_array());
    arrays::array_ref_t<math::normalf_t> normals( shape.attributes().primitive().array( g_N_name));

    boost::uint32_t face_start_index = 0;
    for( int i = 0, e = model.num_faces(); i < e; ++i)
    {
        boost::uint32_t num_verts = verts_per_face[i];

        boost::optional<math::normalf_t> n =  newell_polygon_normal( boost::make_permutation_iterator( points.begin(),
                                                                            face_vert_indices.begin() + face_start_index),
                                                                      boost::make_permutation_iterator( points.begin(),
                                                                            face_vert_indices.begin() + face_start_index + num_verts));

        if( n)
            normals[i] = n.get();

        face_start_index += num_verts;
    }
}

} // geo
} // ramen
