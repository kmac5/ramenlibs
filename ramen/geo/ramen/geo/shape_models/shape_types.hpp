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

#ifndef RAMEN_GEO_SHAPE_MODEL_TYPES_HPP
#define RAMEN_GEO_SHAPE_MODEL_TYPES_HPP

#include<ramen/geo/config.hpp>

namespace ramen
{
namespace geo
{

enum shape_type_t
{
    poly_mesh_shape_k       = 1 << 0,
    subd_mesh_shape_k       = 1 << 1,
    nurbs_curve_shape_k     = 1 << 2,
    nurbs_surface_shape_k   = 1 << 3,
    points_shape_k          = 1 << 4,
    curves_shape_k          = 1 << 5
};

enum shape_mask_t
{
    curves_shape_mask_k = curves_shape_k | nurbs_curve_shape_k,
    nurbs_shape_mask_k  = nurbs_curve_shape_k | nurbs_surface_shape_k,
    mesh_shape_mask_k   = poly_mesh_shape_k | subd_mesh_shape_k
};

enum curve_basis_t
{
    no_basis_k = 0,
    bezier_basis_k,
    bspline_basis_k,
    catmullrom_basis_k,
    hermite_basis_k,
    power_basis_k,
    nurbs_basis_k
};

enum curve_periodicity_t
{
    non_periodic_k,
    periodic_k
};

// For future use...
template<class T>
struct shape_model_traits {};

RAMEN_GEO_API const char *shape_type_to_string( shape_type_t t);

} // geo
} // ramen

#endif
