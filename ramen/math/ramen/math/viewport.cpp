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

#include<ramen/math/viewport.hpp>

namespace ramen
{
namespace math
{

viewport_t::viewport_t() : y_down_( false) {}

float viewport_t::zoom_x() const
{
    return ( device_.size().x)  / ( world_.size().x);
}

float viewport_t::zoom_y() const
{
    return ( device_.size().y)  / ( world_.size().y);
}

point2f_t viewport_t::screen_to_world( const point2i_t& p) const
{
	int y = p.y;

	if( y_down_)
		y = device_.min.y + ( device_.max.y - y);

    return point2f_t( ((p.x - device_.min.x) / zoom_x()) + world_.min.x,
                      ((  y - device_.min.y) / zoom_y()) + world_.min.y);
}

point2i_t viewport_t::world_to_screen( const point2f_t& p) const
{
	int x = ((p.x - world_.min.x) * zoom_x()) + device_.min.x;
	int y = ((p.y - world_.min.y) * zoom_y()) + device_.min.y;

	if( y_down_)
		y = device_.min.y + ( device_.max.y - y);
	
    return point2i_t( x, y);
}

vector2f_t viewport_t::screen_to_world_dir( const vector2f_t& v) const
{
    return vector2f_t( v.x / zoom_x(), v.y / zoom_y());
}

box2f_t viewport_t::screen_to_world( const box2i_t& b) const
{
    return box2f_t( screen_to_world( point2i_t( b.min.x, b.min.y)),
                    screen_to_world( point2i_t( b.max.x, b.max.y)));
}

box2i_t viewport_t::world_to_screen( const box2f_t& b) const
{
    return box2i_t( world_to_screen( point2f_t( b.min.x, b.min.y)),
                    world_to_screen( point2f_t( b.max.x, b.max.y)));
}

void viewport_t::reset()
{
    world_ = box2f_t( point2f_t( device_.min.x, device_.min.y),
                      point2f_t( device_.max.x, device_.max.y));
}

void viewport_t::reset( int w, int h)
{ 
    reset( box2i_t( point2i_t( 0, 0), point2i_t( w - 1, h - 1)));
}

void viewport_t::reset( const box2i_t& device)
{
	device_ = device;
	reset();
}

void viewport_t::reset( const box2i_t& device, const box2f_t& world)
{
	world_ = world;
	device_ = device;
}

void viewport_t::resize( const box2i_t& device)
{
	world_.max.x = world_.min.x + ( device.size().x / zoom_x());
	world_.max.y = world_.min.y + ( device.size().y / zoom_y());
	device_ = device;
}

void viewport_t::resize( int w, int h)
{
    resize( box2i_t( point2i_t( 0, 0), point2i_t( w - 1, h - 1)));
}

void viewport_t::scroll( const point2i_t& inc)
{
	if( y_down_)
        world_.offset_by( vector2f_t( inc.x / zoom_x(), -inc.y / zoom_y()));
	else
        world_.offset_by( vector2f_t( inc.x / zoom_x(),  inc.y / zoom_y()));
}

void viewport_t::scroll_to_center_point( const point2f_t& center)
{
    world_.offset_by( center - world_.center());
}

void viewport_t::zoom( const point2f_t& center, float factor)
{ 
	zoom( center, factor, factor);
}

void viewport_t::zoom( const point2f_t& center, float xfactor, float yfactor)
{
    world_.offset_by( vector2f_t( -center.x, -center.y));
	world_.min.x *= xfactor;
	world_.min.y *= yfactor;
	world_.max.x *= xfactor;
	world_.max.y *= yfactor;
    world_.offset_by( vector2f_t( center.x, center.y));
}
	
matrix33f_t viewport_t::world_to_screen_matrix() const
{
    matrix33f_t m;

    m = matrix33f_t::translation_matrix( vector2f_t( -world_.min.x, -world_.min.y)) *
        matrix33f_t::scale_matrix( vector2f_t( zoom_x(), zoom_y())) *
        matrix33f_t::translation_matrix( vector2f_t( device_.min.x, device_.min.y));

    if( y_down_)
    {
        m = m * matrix33f_t::translation_matrix( vector2f_t( 0, -device_.min.y)) *
                matrix33f_t::scale_matrix( vector2f_t( 1, -1)) *
                matrix33f_t::translation_matrix( vector2f_t( 0, device_.max.y));
    }

    return m;
}

matrix33f_t viewport_t::screen_to_world_matrix() const
{
    matrix33f_t m( world_to_screen_matrix());
    return invert( m).get();
}

} // math
} // ramen
