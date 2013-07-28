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

#ifndef RAMEN_GEO_GLOBAL_NAMES_HPP
#define RAMEN_GEO_GLOBAL_NAMES_HPP

#include<ramen/geo/config.hpp>

#include<ramen/core/name.hpp>

namespace ramen
{
namespace geo
{

RAMEN_GEO_API extern const core::name_t g_P_name;
RAMEN_GEO_API extern const core::name_t g_Pw_name;
RAMEN_GEO_API extern const core::name_t g_N_name;
RAMEN_GEO_API extern const core::name_t g_uv_name;
RAMEN_GEO_API extern const core::name_t g_time_name;
RAMEN_GEO_API extern const core::name_t g_velocity_name;
RAMEN_GEO_API extern const core::name_t g_id_name;
RAMEN_GEO_API extern const core::name_t g_name_name;
RAMEN_GEO_API extern const core::name_t g_material_name;
RAMEN_GEO_API extern const core::name_t g_subd_scheme;

} // geo
} // ramen

#endif
