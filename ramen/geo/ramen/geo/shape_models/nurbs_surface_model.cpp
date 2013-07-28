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

#include<ramen/geo/shape_models/nurbs_surface_model.hpp>

#include<boost/foreach.hpp>

#include<ramen/arrays/array_ref.hpp>

#include<ramen/geo/global_names.hpp>

namespace ramen
{
namespace geo
{

nurbs_surface_model_t::nurbs_surface_model_t( unsigned int udegree,
                                              unsigned int vdegree,
                                              unsigned int upoints,
                                              unsigned int vpoints) : nurbs_model_t(),
                                                                      udegree_( udegree),
                                                                      vdegree_( vdegree),
                                                                      upoints_( upoints),
                                                                      vpoints_( vpoints),
                                                                      uknots_( arrays::array_t( core::float_k)),
                                                                      vknots_( arrays::array_t( core::float_k))
{
    assert( udegree_ > 0);
    assert( vdegree_ > 0);

    assert( upoints_ >= uorder());
    assert( vpoints_ >= vorder());

    u_knots_array().reserve( upoints_ + uorder());
    v_knots_array().reserve( vpoints_ + vorder());
}

unsigned int nurbs_surface_model_t::upoints() const
{
    return upoints_;
}

unsigned int nurbs_surface_model_t::vpoints() const
{
    return vpoints_;
}

unsigned int nurbs_surface_model_t::udegree() const
{
    return udegree_;
}

unsigned int nurbs_surface_model_t::uorder() const
{
    return udegree_ + 1;
}

unsigned int nurbs_surface_model_t::vdegree() const
{
    return vdegree_;
}

unsigned int nurbs_surface_model_t::vorder() const
{
    return vdegree_ + 1;
}

const arrays::array_t& nurbs_surface_model_t::const_u_knots_array() const
{
    return uknots_.read();
}

arrays::array_t& nurbs_surface_model_t::u_knots_array()
{
    return uknots_.write();
}

const arrays::array_t& nurbs_surface_model_t::const_v_knots_array() const
{
    return vknots_.read();
}

arrays::array_t& nurbs_surface_model_t::v_knots_array()
{
    return vknots_.write();
}

const shape_vector_t& nurbs_surface_model_t::const_trim_curves_array() const
{
    return trim_curves_.read();
}

shape_vector_t& nurbs_surface_model_t::trim_curves_array()
{
    return trim_curves_.write();
}

bool nurbs_surface_model_t::check_consistency( const shape_attributes_t& attrs) const
{
    unsigned int num_cvs = 0;

    if( attrs.point().has_attribute( g_Pw_name))
        num_cvs = attrs.point().const_array( g_Pw_name).size();
    else if( attrs.point().has_attribute( g_P_name))
        num_cvs = attrs.point().const_array( g_P_name).size();

    if( num_cvs != ( upoints() * vpoints()))
        return false;

    if( const_u_knots_array().size() != ( upoints() + uorder()))
        return false;

    if( const_v_knots_array().size() != ( vpoints() + vorder()))
        return false;

    // check trim curves.
    BOOST_FOREACH( const shape_t& curve, const_trim_curves_array())
    {
        if( !curve.check_consistency())
            return false;

        // check if the trim curve is not closed, ...
    }

    return true;
}

} // geo
} // ramen
