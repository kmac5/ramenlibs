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

#include<ramen/geo/shape_models/mesh_model.hpp>

#include<numeric>
#include<algorithm>

#include<ramen/arrays/array_ref.hpp>

#include<ramen/geo/shape_attributes.hpp>
#include<ramen/geo/global_names.hpp>

namespace ramen
{
namespace geo
{

mesh_model_t::mesh_model_t() :  verts_per_face_( arrays::array_t( core::uint32_k)),
                                face_vert_indices_( arrays::array_t( core::uint32_k))
{
}

unsigned int mesh_model_t::num_faces() const
{
    return const_verts_per_face_array().size();
}

const arrays::array_t& mesh_model_t::const_verts_per_face_array() const
{
    return verts_per_face_.read();
}

arrays::array_t& mesh_model_t::verts_per_face_array()
{
    return verts_per_face_.write();
}

const arrays::array_t& mesh_model_t::const_face_vert_indices_array() const
{
    return face_vert_indices_.read();
}

arrays::array_t& mesh_model_t::face_vert_indices_array()
{
    return face_vert_indices_.write();
}

bool mesh_model_t::check_consistency( const shape_attributes_t& attrs) const
{
    // TODO: finish this...

    if( !attrs.point().has_attribute( g_P_name))
        return false;

    std::size_t num_points = attrs.point().const_array( g_P_name).size();

    arrays::const_array_ref_t<boost::uint32_t> verts_per_face_ref( const_verts_per_face_array());
    unsigned int num_verts = std::accumulate( verts_per_face_ref.begin(), verts_per_face_ref.end(), 0);

    if( num_verts != face_vert_indices_.read().size())
        return false;

    arrays::const_array_ref_t<boost::uint32_t> face_vert_indices_ref( const_face_vert_indices_array());


    arrays::const_array_ref_t<boost::uint32_t>::const_iterator max_it = std::max_element( face_vert_indices_ref.begin(),
                                                                                          face_vert_indices_ref.end());

    if( *max_it >= num_points)
        return false;

    return true;
}

} // geo
} // ramen
