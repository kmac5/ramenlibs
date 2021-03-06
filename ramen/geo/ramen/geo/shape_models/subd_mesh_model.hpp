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

#ifndef RAMEN_GEO_MODELS_SUBD_MESH_HPP
#define RAMEN_GEO_MODELS_SUBD_MESH_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_models/subd_mesh_model_fwd.hpp>

#include<ramen/geo/shape_models/mesh_model.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API subd_mesh_model_t : public mesh_model_t
{
public:

    subd_mesh_model_t();

    bool check_consistency( const shape_attributes_t& attrs) const;

private:

    core::name_t scheme_;
};

template<>
struct shape_model_traits<subd_mesh_model_t>
{
    static shape_type_t type() { return subd_mesh_shape_k;}
};

} // geo
} // ramen

#endif
