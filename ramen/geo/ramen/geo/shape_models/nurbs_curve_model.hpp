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

#ifndef RAMEN_GEO_MODELS_NURBS_CURVE_MODEL_HPP
#define RAMEN_GEO_MODELS_NURBS_CURVE_MODEL_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_models/nurbs_model.hpp>

#include<ramen/core/copy_on_write.hpp>

#include<ramen/arrays/array_ref.hpp>

#include<ramen/geo/shape_models/nurbs_curve_model_fwd.hpp>
#include<ramen/geo/shape_attributes_fwd.hpp>
#include<ramen/geo/shape_fwd.hpp>

#include<ramen/geo/shape_models/shape_types.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API nurbs_curve_model_t : public nurbs_model_t
{
public:

    explicit nurbs_curve_model_t( unsigned int degree);

    unsigned int degree() const;
    unsigned int order() const;

    const arrays::array_t& const_knots_array() const;
    arrays::array_t& knots_array();

    bool check_consistency( const shape_attributes_t& attrs) const;

private:

    unsigned int degree_;
    core::copy_on_write_t<arrays::array_t> knots_;
};

template<>
struct shape_model_traits<nurbs_curve_model_t>
{
    static shape_type_t type() { return nurbs_curve_shape_k;}
};

} // geo
} // ramen

#endif
