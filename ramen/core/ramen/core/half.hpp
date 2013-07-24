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

#ifndef RAMEN_CORE_HALF_HPP
#define RAMEN_CORE_HALF_HPP

#include<ramen/core/config.hpp>

#include<OpenEXR/halfLimits.h>

#include<boost/type_traits/is_float.hpp>
#include<boost/type_traits/is_floating_point.hpp>

#include<boost/type_traits/detail/bool_trait_def.hpp>

// spec boost type traits
namespace boost
{

//* is a type T a floating-point type described in the standard (3.9.1p8)
BOOST_TT_AUX_BOOL_TRAIT_CV_SPEC1( is_float,half,true)
BOOST_TT_AUX_BOOL_TRAIT_CV_SPEC1( is_floating_point,half,true)

} // namespace boost

#include <boost/type_traits/detail/bool_trait_undef.hpp>

#endif
