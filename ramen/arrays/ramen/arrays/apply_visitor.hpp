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

#ifndef RAMEN_ARRAYS_APPLY_VISITOR_HPP
#define RAMEN_ARRAYS_APPLY_VISITOR_HPP

#include<ramen/arrays/config.hpp>

#include<ramen/core/exceptions.hpp>

#include<ramen/arrays/array_ref.hpp>

namespace ramen
{
namespace arrays
{
namespace detail
{

template<class T, class Visitor>
void apply_visitor( Visitor& v, array_t& array)
{
    array_ref_t<T> array_ref( array);
    v( array_ref);
}

template<class T, class Visitor>
void apply_visitor( Visitor& v, const array_t& array)
{
    const_array_ref_t<T> array_ref( array);
    v( array_ref);
}

} // detail

#define RAMEN_ARRAYS_APPLY_VISITOR_CASE( type_enum_k, type_name)\
    case type_enum_k:\
        detail::apply_visitor<type_name>( v, array);\
    break;

template<class Visitor>
void apply_visitor( Visitor& v, array_t& array)
{
    switch( array.type())
    {
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::bool_k, bool)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::float_k, float)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::double_k, double)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int8_k, boost::int8_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint8_k, boost::uint8_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int16_k, boost::int16_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint16_k, boost::uint16_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int32_k, boost::int32_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint32_k, boost::uint32_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int64_k, boost::int64_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint64_k, boost::uint64_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point2i_k, math::point2i_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point2f_k, math::point2f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point3f_k, math::point3f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector2i_k, math::vector2i_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector2f_k, math::vector2f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector3f_k, math::vector3f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::normalf_k, math::normalf_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::hpoint2f_k, math::hpoint2f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::hpoint3f_k, math::hpoint3f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::box2i_k, math::box2i_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::color3f_k, color::color3f_t)

        #if RAMEN_WITH_HALF
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::half_k, half)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point2h_k, math::point2h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point3h_k, math::point3h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector2h_k, math::vector2h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector3h_k, math::vector3h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::normalh_k, math::normalh_t)
        #endif

        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::string8_k, core::string8_t)

        default:
            throw core::not_implemented();
    };
}

template<class Visitor>
void apply_visitor( Visitor& v, const array_t& array)
{
    switch( array.type())
    {
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::bool_k, bool)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::float_k, float)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::double_k, double)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int8_k, boost::int8_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint8_k, boost::uint8_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int16_k, boost::int16_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint16_k, boost::uint16_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int32_k, boost::int32_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint32_k, boost::uint32_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::int64_k, boost::int64_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::uint64_k, boost::uint64_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point2i_k, math::point2i_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point2f_k, math::point2f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point3f_k, math::point3f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector2i_k, math::vector2i_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector2f_k, math::vector2f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector3f_k, math::vector3f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::normalf_k, math::normalf_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::hpoint2f_k, math::hpoint2f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::hpoint3f_k, math::hpoint3f_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::box2i_k, math::box2i_t)
        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::color3f_k, color::color3f_t)

        #if RAMEN_WITH_HALF
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::half_k, half)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point2h_k, math::point2h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::point3h_k, math::point3h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector2h_k, math::vector2h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::vector3h_k, math::vector3h_t)
            RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::normalh_k, math::normalh_t)
        #endif

        RAMEN_ARRAYS_APPLY_VISITOR_CASE( core::string8_k, core::string8_t)

        default:
            throw core::not_implemented();
    };
}

#undef RAMEN_ARRAYS_APPLY_VISITOR_CASE

} // arrays
} // ramen

#endif
