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

#ifndef RAMEN_GEO_MODELS_NURBS_SURFACE_MODEL_HPP
#define RAMEN_GEO_MODELS_NURBS_SURFACE_MODEL_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_models/nurbs_surface_model_fwd.hpp>

#include<ramen/geo/shape_models/nurbs_model.hpp>

#include<ramen/core/copy_on_write.hpp>

#include<ramen/arrays/array_ref.hpp>

#include<ramen/geo/shape_models/shape_types.hpp>
#include<ramen/geo/shape_attributes_fwd.hpp>
#include<ramen/geo/shape_vector.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API nurbs_surface_model_t : public nurbs_model_t
{
public:

    nurbs_surface_model_t( unsigned int udegree, unsigned int vdegree,
                            unsigned int upoints, unsigned int vpoints);

    unsigned int upoints() const;
    unsigned int vpoints() const;

    unsigned int udegree() const;
    unsigned int uorder() const;

    unsigned int vdegree() const;
    unsigned int vorder() const;

    const arrays::array_t& const_u_knots_array() const;
    arrays::array_t& u_knots_array();

    const arrays::array_t& const_v_knots_array() const;
    arrays::array_t& v_knots_array();

    const shape_vector_t& const_trim_curves_array() const;
    shape_vector_t& trim_curves_array();

    bool check_consistency( const shape_attributes_t& attrs) const;

private:

    unsigned int udegree_, vdegree_;
    unsigned int upoints_, vpoints_;
    core::copy_on_write_t<arrays::array_t> uknots_;
    core::copy_on_write_t<arrays::array_t> vknots_;

    core::copy_on_write_t<shape_vector_t> trim_curves_;
};

template<>
struct shape_model_traits<nurbs_surface_model_t>
{
    static shape_type_t type() { return nurbs_surface_shape_k;}
};

} // geo
} // ramen

#endif
