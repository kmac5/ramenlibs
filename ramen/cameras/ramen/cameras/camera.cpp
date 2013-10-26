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

#include<ramen/cameras/camera.hpp>

#include<cassert>

#include<ramen/cameras/distortion_function.hpp>

namespace ramen
{
namespace cameras
{

camera_t::camera_t() {}
camera_t::~camera_t() {}

void camera_t::set_viewport_size( std::size_t w, std::size_t h)
{
    assert( w != 0);
    assert( h != 0);

    width_ = w;
    height_ = h;
    aspect_ = static_cast<float>( width_) / height_;
}

bool camera_t::is_linear() const
{
    return do_is_linear();
}

bool camera_t::do_is_linear() const
{
    return false;
}

} // cameras
} // ramen
