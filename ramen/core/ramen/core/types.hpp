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

#ifndef RAMEN_CORE_TYPES_HPP
#define RAMEN_CORE_TYPES_HPP

#include<ramen/core/config.hpp>

#include<boost/cstdint.hpp>
#include<boost/mpl/bool.hpp>

#include<ramen/math/hpoint2.hpp>
#include<ramen/math/hpoint3.hpp>
#include<ramen/math/normal.hpp>

#include<ramen/math/box2.hpp>

#include<ramen/color/color3.hpp>

#include<ramen/core/string8.hpp>

#ifdef RAMEN_WITH_HALF
    #include<ramen/core/half.hpp>
#endif

namespace ramen
{
namespace core
{

enum type_t
{
    bool_k,
    float_k,
    double_k,
    int8_k,
    uint8_k,
    int16_k,
    uint16_k,
    int32_k,
    uint32_k,
    int64_k,
    uint64_k,
    point2i_k,
    point2f_k,
    point3f_k,
    vector2i_k,
    vector2f_k,
    vector3f_k,
    normalf_k,

    hpoint2f_k,
    hpoint3f_k,

    box2i_k,

    color3f_k,

    #ifdef RAMEN_WITH_HALF
        half_k,
        point2h_k,
        point3h_k,
        vector2h_k,
        vector3h_k,
        normalh_k,
    #endif

    string8_k
};

template<class T>
struct type_traits {};

#define RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( type_name, type_enum_k)\
    template<>\
    struct type_traits<type_name>\
    {\
        static type_t type() { return type_enum_k;}\
        static type_name default_value() { return type_name( 0);}\
    }

RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( bool, bool_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( float, float_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( double, double_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::int8_t, int8_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::uint8_t, uint8_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::int16_t, int16_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::uint16_t, uint16_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::int32_t, int32_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::uint32_t, uint32_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::int64_t, int64_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( boost::uint64_t, uint64_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::point2i_t, point2i_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::point2f_t, point2f_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::point3f_t, point3f_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::vector2i_t, vector2i_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::vector2f_t, vector2f_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::vector3f_t, vector3f_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::normalf_t, normalf_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::hpoint2f_t, hpoint2f_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::hpoint3f_t, hpoint3f_k);
RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( color::color3f_t, color3f_k);

// special cases for boxes
template<>
struct type_traits<math::box2i_t>
{
    static type_t type() { return box2i_k;}
    static math::box2i_t default_value() { return math::box2i_t();}
};

#ifdef RAMEN_WITH_HALF
    RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( half, half_k);
    RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::point2h_t, point2h_k);
    RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::point3h_t, point3h_k);
    RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::vector2h_t, vector2h_k);
    RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::vector3h_t, vector3h_k);
    RAMEN_ARRAY_TYPE_TRAITS_SPECIALIZATION( math::normalh_t, normalh_k);
#endif

// special case for strings
template<>
struct type_traits<string8_t>
{
    static type_t type() { return string8_k;}
    static string8_t default_value() { return string8_t();}
};

RAMEN_CORE_API const char *type_to_string( type_t t);

} // core
} // ramen

#endif
