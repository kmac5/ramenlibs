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

#include<ramen/geo/visitors/triangulate.hpp>

#include<ramen/geo/shape.hpp>
#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>

#include<ramen/geo/global_names.hpp>

namespace ramen
{
namespace geo
{

triangulate_visitor::triangulate_visitor( bool keep_quads) : shape_visitor()
{
}

void triangulate_visitor::visit( poly_mesh_model_t& model, shape_t& shape)
{
    if( !shape.attributes().point().has_attribute( g_P_name))
        throw core::runtime_error( core::string8_t( "No P attribute found in triangulate visitor"));

    shape_t new_shape( shape_t::create_poly_mesh());
    poly_mesh_model_t& new_model( shape_cast<poly_mesh_model_t>( new_shape));
    do_visit( model, shape, new_model, new_shape);
}

void triangulate_visitor::visit( subd_mesh_model_t& model, shape_t& shape)
{
    if( !shape.attributes().point().has_attribute( g_P_name))
        throw core::runtime_error( core::string8_t( "No P attribute found in triangulate visitor"));

    shape_t new_shape( shape_t::create_subd_mesh());
    subd_mesh_model_t& new_model( shape_cast<subd_mesh_model_t>( new_shape));
    do_visit( model, shape, new_model, new_shape);
}

void triangulate_visitor::do_visit( mesh_model_t& model,
                                    shape_t& shape,
                                    mesh_model_t& new_model,
                                    shape_t& new_shape)
{
    // copy attributes
    new_shape.attributes().point()     = shape.attributes().point();
    new_shape.attributes().constant()  = shape.attributes().constant();
    new_shape.attributes().vertex()    = attribute_table_t::create_empty_with_same_attributes( shape.attributes().vertex());
    new_shape.attributes().primitive() = attribute_table_t::create_empty_with_same_attributes( shape.attributes().primitive());

    arrays::const_array_ref_t<math::point3f_t> points( shape.attributes().point().const_array( g_P_name));
    arrays::const_array_ref_t<boost::uint32_t> verts_per_face( model.const_verts_per_face_array());
    arrays::const_array_ref_t<boost::uint32_t> face_vert_indices( model.const_face_vert_indices_array());

    boost::uint32_t face_start_index = 0;
    for( int i = 0, e = model.num_faces(); i < e; ++i)
    {
        boost::uint32_t num_verts = verts_per_face[i];
        if( num_verts == 3)
            copy_face( model, shape, i, face_start_index, new_model, new_shape);
        else
        {
            if( num_verts == 4)
            {
                if( keep_quads_)
                    copy_face( model, shape, i, face_start_index, new_model, new_shape);
                else
                    triangulate_quad( model, shape, i, face_start_index, new_model, new_shape);
            }
            else
            {
                // TODO: triangulate polygon here
                throw core::not_implemented();
            }
        }

        face_start_index += num_verts;
    }

    shape.swap( new_shape);
    throw core::not_implemented();
}

void triangulate_visitor::copy_face( mesh_model_t& model,
                                     const shape_t& shape,
                                     std::size_t index,
                                     boost::uint32_t face_start_index,
                                     mesh_model_t& new_model,
                                     shape_t& new_shape)
{
    arrays::const_array_ref_t<boost::uint32_t> verts_per_face( model.const_verts_per_face_array());
    arrays::const_array_ref_t<boost::uint32_t> face_vert_indices( model.const_face_vert_indices_array());

    arrays::array_ref_t<boost::uint32_t> new_verts_per_face( new_model.verts_per_face_array());
    arrays::array_ref_t<boost::uint32_t> new_face_vert_indices( new_model.face_vert_indices_array());

    int num_verts = verts_per_face[index];
    new_verts_per_face.push_back( num_verts);
    for( int i = 0; i < num_verts; ++i)
    {
        int vertex_num = face_vert_indices[face_start_index +i];
        new_face_vert_indices.push_back( vertex_num);
        new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                         vertex_num);
    }

    // copy face attributes
    new_shape.attributes().primitive().push_back_attribute_values_copy( shape.attributes().primitive(),
                                                                        index);
}

void triangulate_visitor::triangulate_quad( mesh_model_t& model,
                                            const shape_t& shape,
                                            std::size_t index,
                                            boost::uint32_t face_start_index,
                                            mesh_model_t& new_model,
                                            shape_t& new_shape)
{
    arrays::const_array_ref_t<boost::uint32_t> verts_per_face( model.const_verts_per_face_array());
    arrays::const_array_ref_t<boost::uint32_t> face_vert_indices( model.const_face_vert_indices_array());

    arrays::array_ref_t<boost::uint32_t> new_verts_per_face( new_model.verts_per_face_array());
    arrays::array_ref_t<boost::uint32_t> new_face_vert_indices( new_model.face_vert_indices_array());

    assert( verts_per_face[index] == 4);

    // First triangle
    new_verts_per_face.push_back( 3);
    new_face_vert_indices.push_back( face_vert_indices[face_start_index]);
    new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                     face_start_index);

    new_face_vert_indices.push_back( face_vert_indices[face_start_index + 1]);
    new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                     face_start_index + 1);

    new_face_vert_indices.push_back( face_vert_indices[face_start_index + 3]);
    new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                     face_start_index + 3);
    
    new_shape.attributes().primitive().push_back_attribute_values_copy( shape.attributes().primitive(),
                                                          index);
    // Second triangle
    new_verts_per_face.push_back( 3);
    new_face_vert_indices.push_back( face_vert_indices[face_start_index + 1]);
    new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                     face_start_index + 1);

    new_face_vert_indices.push_back( face_vert_indices[face_start_index + 2]);
    new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                     face_start_index + 2);
    
    new_face_vert_indices.push_back( face_vert_indices[face_start_index + 3]);
    new_shape.attributes().vertex().push_back_attribute_values_copy( shape.attributes().vertex(),
                                                                     face_start_index + 3);

    new_shape.attributes().primitive().push_back_attribute_values_copy( shape.attributes().primitive(),
                                                                        index);    
}

} // geo
} // ramen
