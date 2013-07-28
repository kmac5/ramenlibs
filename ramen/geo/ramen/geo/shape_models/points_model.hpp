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

#ifndef RAMEN_GEO_MODELS_POINTS_MODEL_HPP
#define RAMEN_GEO_MODELS_POINTS_MODEL_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_models/points_model_fwd.hpp>
#include<ramen/geo/shape_models/visitable.hpp>

#include<ramen/geo/shape_attributes_fwd.hpp>
#include<ramen/geo/shape_models/shape_types.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API points_model_t : public visitable_t
{
public:

    points_model_t();

    bool check_consistency( const shape_attributes_t& attrs) const;
};

template<>
struct shape_model_traits<points_model_t>
{
    static shape_type_t type() { return points_shape_k;}
};

} // geo
} // ramen

#endif
