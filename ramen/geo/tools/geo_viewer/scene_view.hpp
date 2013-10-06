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

#ifndef GEO_VIEWER_SCENE_VIEW_HPP
#define GEO_VIEWER_SCENE_VIEW_HPP

#include<geo_viewer/scene_view_fwd.hpp>

#include<boost/filesystem/path.hpp>

#include<ramen/gl/glew.hpp>

#include<QtOpenGL/QGLWidget>

class QMouseEvent;

class scene_view_t : public QGLWidget
{
    Q_OBJECT

public:

    scene_view_t( QWidget *parent = 0);

    void load_scene( const boost::filesystem::path& p);
    
protected:

    virtual void initializeGL();
    virtual void resizeGL( int w, int h);
    virtual void paintGL();

    virtual void mousePressEvent( QMouseEvent *event);
    virtual void mouseMoveEvent( QMouseEvent *event);
    virtual void mouseReleaseEvent( QMouseEvent *event);

private:

    void draw_grid() const;
    void draw_world_axes() const;
};

#endif
