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

#ifndef RAMEN_MATH_CONSTANTS_HPP
#define RAMEN_MATH_CONSTANTS_HPP

#include<ramen/math/config.hpp>

namespace ramen
{
namespace math
{

template<class T> struct constants {};

template<> struct constants<float>
{
    static float pi()		{ return 3.141592653589793238462643383279502884197f;}
    static float half_pi()	{ return 1.570796326794896619231321691639751442098f;}
    static float double_pi(){ return 6.283185307179586476925286766559005768394f;}
    static float pi4()		{ return 12.56637061435917295385057353311801153678f;}
    static float deg2rad()	{ return 0.017453292519943295769236907684886127134f;}
    static float rad2deg()	{ return 57.29577951308232087679815481410517033381f;}
    static float sqrt2()	{ return 1.414213562373095048801688724209698078569f;}
    static float sqrt3()	{ return 1.732050807568877293527446341505872366942f;}
};

template<> struct constants<double>
{
    static double pi()		 { return 3.141592653589793238462643383279502884197;}
    static double half_pi()	 { return 1.570796326794896619231321691639751442098;}
    static double double_pi(){ return 6.283185307179586476925286766559005768394;}
    static double pi4()		 { return 12.56637061435917295385057353311801153678;}
    static double deg2rad()	 { return 0.017453292519943295769236907684886127134;}
    static double rad2deg()	 { return 57.29577951308232087679815481410517033381;}
    static double sqrt2()	 { return 1.414213562373095048801688724209698078570;}
    static double sqrt3()	 { return 1.732050807568877293527446341505872366940;}
};

} // math
} // ramen

#endif
