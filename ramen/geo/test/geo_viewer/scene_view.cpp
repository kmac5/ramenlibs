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

#include<geo_viewer/scene_view.hpp>

#include<ramen/gl/gl.hpp>

#include<iostream>

using namespace ramen;

scene_view_t::scene_view_t( QWidget *parent) : QGLWidget( parent)
{
}

void scene_view_t::initializeGL()
{
    if( GLenum err = glewInit() != GLEW_OK)
    {
        std::cerr << "Could not init GLEW, error = " << glewGetErrorString( err) << ". Exiting." << std::endl;
        std::abort();
    }

    gl::clear_errors();

    gl::clear_color( 0.35, 0.35, 0.35, 0.0);
}

void scene_view_t::resizeGL( int w, int h)
{
}

void scene_view_t::paintGL()
{
    gl::clear( GL_COLOR_BUFFER_BIT);
}

void scene_view_t::draw_grid() const
{
    glColor3f( .8f, .8f, .8f);
    glBegin( GL_LINES);
        for( int i = 0; i < 40; ++i)
        {
            glVertex3f( -20, 0, -20 + i);
            glVertex3f(  20, 0, -20 + i);

            glVertex3f( -20 + i, 0, -20);
            glVertex3f( -20 + i, 0,  20);
        }
    glEnd();

    glColor3f( .4f, .4f, .4f);
    glBegin( GL_LINES);
        for( int i = 0; i < 40; ++i)
        {
            glVertex3f( -20.5, 0, -20.5 + i);
            glVertex3f(  20.5, 0, -20.5 + i);

            glVertex3f( -20.5 + i, 0, -20.5);
            glVertex3f( -20.5 + i, 0,  20.5);
        }
    glEnd();
}
