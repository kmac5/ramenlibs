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

#ifndef RAMEN_CAMERAS_CAMERA_HPP
#define RAMEN_CAMERAS_CAMERA_HPP

#include<ramen/cameras/camera_fwd.hpp>

#include<ramen/math/matrix44.hpp>

namespace ramen
{
namespace cameras
{

/*!
\ingroup cameras
\brief Base camera class.
*/
class RAMEN_CAMERAS_API camera_t
{
public:
    
    virtual ~camera_t();
    
protected:

    camera_t();
    
private:

    // non-copyable
    camera_t( const camera_t&);
    camera_t& operator=( const camera_t&);

    math::point3d_t eye_;
};

} // cameras
} // ramen

#endif