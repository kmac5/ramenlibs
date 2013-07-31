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

#ifndef RAMEN_MATH_VIEWPORT_HPP
#define	RAMEN_MATH_VIEWPORT_HPP

#include<ramen/math/config.hpp>

#include<ramen/math/box2.hpp>
#include<ramen/math/matrix33.hpp>

namespace ramen
{
namespace math
{

class RAMEN_MATH_API viewport_t
{
public:

    viewport_t();

    bool y_down() const         { return y_down_;}
    void set_y_down( bool b)    { y_down_ = b;}

    const box2f_t& world() const   { return world_;}
    const box2i_t& device() const  { return device_;}

    float zoom_x() const;
    float zoom_y() const;

    point2f_t screen_to_world( const point2i_t& p) const;
    point2i_t world_to_screen( const point2f_t& p) const;

    vector2f_t screen_to_world_dir( const vector2f_t& v) const;

    box2f_t screen_to_world( const box2i_t& b) const;
    box2i_t world_to_screen( const box2f_t& b) const;

    matrix33f_t world_to_screen_matrix() const;
    matrix33f_t screen_to_world_matrix() const;

    void reset();
    void reset( int w, int h);
    void reset( const box2i_t& device);
    void reset( const box2i_t& device, const box2f_t& world);

    void resize( const box2i_t& device);
    void resize( int w, int h);

    void scroll( const vector2i_t& inc);
    void scroll_to_center_point( const point2f_t& center);

    void zoom( const point2f_t& center, float factor);
    void zoom( const point2f_t& center, float xfactor, float yfactor);

private:

    box2i_t device_;
    box2f_t world_;
    bool y_down_;
};

} // math
} // ramen

#endif
