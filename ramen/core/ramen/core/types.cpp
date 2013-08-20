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

#include<ramen/core/types.hpp>

#include<cassert>

namespace ramen
{
namespace core
{

const char *type_to_string( type_t t)
{
    #define RAMEN_ARRAYS_TYPE2STRING_CASE( type_enum_k, type_name)\
        case type_enum_k:\
            return type_name;

    switch( t)
    {
        RAMEN_ARRAYS_TYPE2STRING_CASE( bool_k, "bool")
        RAMEN_ARRAYS_TYPE2STRING_CASE( float_k, "float")
        RAMEN_ARRAYS_TYPE2STRING_CASE( double_k, "double")
        RAMEN_ARRAYS_TYPE2STRING_CASE( int8_k, "int8_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( uint8_k, "uint8_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( int16_k, "int16_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( uint16_k, "uint16_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( int32_k, "int32_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( uint32_k, "uint32_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( point2i_k, "point2i_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( point2f_k, "point2f_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( point3f_k, "point3f_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( vector2i_k, "vector2i_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( vector2f_k, "vector2f_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( vector3f_k, "vector3f_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( normalf_k, "normalf_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( hpoint2f_k, "hpoint2f_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( hpoint3f_k, "hpoint3f_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( box2i_k, "box2i_t")
        RAMEN_ARRAYS_TYPE2STRING_CASE( color3f_k, "color3f_t")

        #ifdef RAMEN_WITH_HALF
            RAMEN_ARRAYS_TYPE2STRING_CASE( half_k ,"half")
            RAMEN_ARRAYS_TYPE2STRING_CASE( point2h_k, "point2h_t")
            RAMEN_ARRAYS_TYPE2STRING_CASE( point3h_k, "point3h_t")
            RAMEN_ARRAYS_TYPE2STRING_CASE( vector2h_k, "vector2h_t")
            RAMEN_ARRAYS_TYPE2STRING_CASE( vector3h_k, "vector3h_t")
            RAMEN_ARRAYS_TYPE2STRING_CASE( normalh_k, "normalh_t")
        #endif

        RAMEN_ARRAYS_TYPE2STRING_CASE( string8_k, "string8_t")

        default:
            assert( false && "Unknown type in type_to_string");
            return 0;
    }

    #undef RAMEN_ARRAYS_TYPE2STRING_CASE

    // for dumb compilers...
    return 0;
}

} // core
} // ramen
