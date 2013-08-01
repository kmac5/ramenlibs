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

#ifndef RAMEN_COLOR_COLORSPACE_HPP
#define RAMEN_COLOR_COLORSPACE_HPP

#include<ramen/color/config.hpp>

#include<boost/mpl/int.hpp>

namespace ramen
{
namespace color
{

struct rgb_t
{
    typedef boost::mpl::int_<3> size_type;
};

struct xyz_t
{
    typedef boost::mpl::int_<3> size_type;
};

struct hsv_t
{
    typedef boost::mpl::int_<3> size_type;
};

struct rgba_t
{
    typedef boost::mpl::int_<4> size_type;
};

} // color
} // ramen

#endif
