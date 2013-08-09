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

#ifndef RAMEN_QWIDGETS_TIME_SCALE_HPP
#define	RAMEN_QWIDGETS_TIME_SCALE_HPP

#include<ramen/qwidgets/config.hpp>

#include<QWidget>

namespace ramen
{
namespace qwidgets
{

class RAMEN_QWIDGETS_API time_scale_t : public QWidget
{
    Q_OBJECT

public:

    time_scale_t( QWidget *parent = 0);

    void setRange( int lo, int hi);
    void setMinimum( int m);
    void setMaximum( int m);

    void setValue( double v);

protected:

    virtual void mouseMoveEvent( QMouseEvent *event);
    virtual void mousePressEvent( QMouseEvent *event);
    virtual void mouseReleaseEvent( QMouseEvent *event);
    virtual void paintEvent ( QPaintEvent *event);

Q_SIGNALS:

    void valueChanged( double value);

private:

    int round_halfup( float x) const;
    int frame_from_mouse_pos( int x) const;

    int min_value_;
    int max_value_;
    double value_;

    int last_x_;
    bool drag_;
};

} // qwidgets
} // ramen

#endif
