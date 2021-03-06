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

#include<ramen/geo/shape_models/shape_types.hpp>

#include<cassert>

namespace ramen
{
namespace geo
{

const char *shape_type_to_string( shape_type_t t)
{
    switch( t)
    {
        case poly_mesh_shape_k:
            return "poly_mesh";

        case subd_mesh_shape_k:
            return "subd_mesh";

        case nurbs_curve_shape_k:
            return "nurbs_curve";

        case nurbs_surface_shape_k:
            return "nurbs_surface";

        case points_shape_k:
            return "points";

        case curves_shape_k:
            return "curves";

        default:
            assert( false && "Unknown shape type in shape_type_to_string");
    }

    // for dumb compilers.
    return 0;
}

} // geo
} // ramen
