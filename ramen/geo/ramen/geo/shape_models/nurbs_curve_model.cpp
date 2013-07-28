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

#include<ramen/geo/shape_models/nurbs_curve_model.hpp>

#include<ramen/arrays/array_ref.hpp>

#include<ramen/geo/shape_attributes.hpp>
#include<ramen/geo/global_names.hpp>

namespace ramen
{
namespace geo
{

nurbs_curve_model_t::nurbs_curve_model_t( unsigned int degree) : nurbs_model_t(),
                                                                 degree_( degree),
                                                                 knots_( arrays::array_t( core::float_k))
{
    assert( degree_ > 0);
}

unsigned int nurbs_curve_model_t::degree() const
{
    return degree_;
}

unsigned int nurbs_curve_model_t::order() const
{
    return degree_ + 1;
}

const arrays::array_t& nurbs_curve_model_t::const_knots_array() const
{
    return knots_.read();
}

arrays::array_t& nurbs_curve_model_t::knots_array()
{
    return knots_.write();
}

bool nurbs_curve_model_t::check_consistency( const shape_attributes_t& attrs) const
{
    unsigned int num_cvs = 0;

    if( attrs.point().has_attribute( g_Pw_name))
        num_cvs = attrs.point().const_array( g_Pw_name).size();
    else if( attrs.point().has_attribute( g_P_name))
        num_cvs = attrs.point().const_array( g_P_name).size();

    if( num_cvs == 0)
        return false;

    if( const_knots_array().size() != ( num_cvs + order()))
        return false;

    return true;
}

} // geo
} // ramen
