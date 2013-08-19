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

#include<ramen/math/point3.hpp>
#include<ramen/math/matrix44.hpp>
#include<ramen/math/constants.hpp>

#include<ramen/gl/gl.hpp>

#include<iostream>

#include<QMouseEvent>

using namespace ramen;

namespace
{

class camera_t
{
public:

    camera_t()
    {
        set_viewport_size( 640, 480);
        eye_ = math::point3f_t( 0, 0, 0);
        u_   = math::vector3f_t( 1, 0, 0);
        v_   = math::vector3f_t( 0, 1, 0);
        n_   = math::vector3f_t( 0, 0, 1);
        fov_ = 45.0f * math::constants<float>::deg2rad();
        near_clip_ = 0.01f;
        far_clip_ = 10000.0f;
    }

    void set_viewport_size( int w, int h)
    {
        width_ = w;
        height_ = h;
        aspect_ = static_cast<float>( width_) / height_;
    }

    void setup_camera_transform()
    {
        gluLookAt( eye_.x, eye_.y, eye_.z,
                   eye_.x + n_.x, eye_.y + n_.y, eye_.z + n_.z,
                   v_.x, v_.y, v_.z);
    }

    void setup_projection()
    {
        glMatrixMode( GL_PROJECTION);
        glLoadIdentity();
        gluPerspective( fov_, aspect_, near_clip_, far_clip_);
        glMatrixMode( GL_MODELVIEW);
    }

    void track( const math::vector2f_t& d)
    {
        eye_ += d.x * u_;
        eye_ += d.y * v_;
    }

    void dolly( float d)
    {
        eye_ += d * n_;
    }

    int width_, height_;
    float aspect_;
    math::point3f_t eye_;
    math::vector3f_t u_, v_, n_;
    float fov_;
    float near_clip_;
    float far_clip_;
};

camera_t g_camera;

struct maya_like_camera_controller_t
{
    enum movement_type_t
    {
        none_k = 0,
        tumble_k,
        track_k,
        dolly_k
    };

    maya_like_camera_controller_t()
    {
        camera_ = 0;
        press_pos_ = math::point2i_t( 0, 0);
        move_type_ = none_k;
    }

    void mouse_press( const math::point2i_t& pos, movement_type_t m_type)
    {
        move_type_ = m_type;
        press_pos_ = pos;
        last_pos_ = pos;
    }

    void mouse_move( const math::point2i_t& pos)
    {
        switch( move_type_)
        {
            case track_k:
            {
                math::vector2f_t d( 0.001f * ( pos.x - last_pos_.x),
                                    0.001f * ( pos.y - last_pos_.y));
                camera_->track( d);
            }
            break;

            case dolly_k:
                camera_->dolly( ( pos.x - last_pos_.x) * 0.01f);
            break;
        }

        last_pos_ = pos;
    }

    void mouse_release( const math::point2i_t& pos)
    {
        move_type_ = none_k;
    }

    camera_t *camera_;
    movement_type_t move_type_;
    math::point2i_t press_pos_;
    math::point2i_t last_pos_;
};

maya_like_camera_controller_t g_cam_controller;

} // unnamed

scene_view_t::scene_view_t( QWidget *parent) : QGLWidget( parent) {}

void scene_view_t::initializeGL()
{
    if( GLenum err = glewInit() != GLEW_OK)
    {
        std::cerr << "Could not init GLEW, error = " << glewGetErrorString( err)
                  << ". Exiting." << std::endl;
        std::abort();
    }

    if( !GLEW_VERSION_2_0)
    {
        std::cerr << "OpenGL 2.0 or newer not found. Exiting" << std::endl;
        std::abort();
    }

    gl::clear_errors();
    glClearColor( 0.35, 0.35, 0.35, 0.0);

    g_cam_controller.camera_ = &g_camera;
}

void scene_view_t::resizeGL( int w, int h)
{
    glViewport( 0, 0, w, h);
    g_camera.set_viewport_size( w, h);
}

void scene_view_t::paintGL()
{
    glClear( GL_COLOR_BUFFER_BIT);

    g_camera.setup_projection();

    glLoadIdentity();
    g_camera.setup_camera_transform();

    draw_grid();
    draw_world_axes();

    /*
    glLineWidth( 1.0f);
    glColor3f( 1.0f, 1.0f, 1.0f);
    glBegin( GL_LINE_LOOP);
        glVertex3f( -3,  3, 1000);
        glVertex3f(  3,  3, 1000);
        glVertex3f( -3, -3, 1000);
        glVertex3f(  3, -3, 1000);
    glEnd();
    */
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

    glColor3f( .0f, .0f, .0f);
    glLineWidth( 2.0f);
    glBegin( GL_LINES);
        glVertex3f( -20, 0, 0);
        glVertex3f(  20, 0, 0);

        glVertex3f( 0, 0, -20);
        glVertex3f( 0, 0,  20);
    glEnd();
}

void scene_view_t::draw_world_axes() const
{
    const int axis_lenght = 100;
    glLineWidth( 4.0f);
    glColor3f( 1.0, 0.0f, 0.0f);
    glBegin( GL_LINE);
        glVertex3f( 0, 0, 0);
        glVertex3f( axis_lenght, 0, 0);
    glEnd();

    glColor3f( 0.0, 1.0f, 0.0f);
    glBegin( GL_LINE);
        glVertex3f( 0, 0, 0);
        glVertex3f( 0, axis_lenght, 0);
    glEnd();

    glColor3f( 0.0, 0.0f, 1.0f);
    glBegin( GL_LINE);
        glVertex3f( 0, 0, 0);
        glVertex3f( 0, 0, axis_lenght);
    glEnd();
}

void scene_view_t::mousePressEvent( QMouseEvent *event)
{
    math::point2i_t pos( event->x(), event->y());
    maya_like_camera_controller_t::movement_type_t m_type;

    if( event->modifiers() & Qt::ShiftModifier)
        m_type = maya_like_camera_controller_t::track_k;
    else
    {
        if( event->modifiers() & Qt::AltModifier)
            m_type = maya_like_camera_controller_t::dolly_k;
        else
            m_type = maya_like_camera_controller_t::tumble_k;
    }

    g_cam_controller.mouse_press( pos, m_type);
    event->accept();
}

void scene_view_t::mouseMoveEvent( QMouseEvent *event)
{
    g_cam_controller.mouse_move( math::point2i_t( event->x(), event->y()));
    event->accept();
    update();
}

void scene_view_t::mouseReleaseEvent( QMouseEvent *event)
{
    g_cam_controller.mouse_release( math::point2i_t( event->x(), event->y()));
    event->accept();
}
