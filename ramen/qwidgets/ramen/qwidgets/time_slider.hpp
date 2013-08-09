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

#ifndef RAMEN_QWIDGETS_TIME_SLIDER_HPP
#define	RAMEN_QWIDGETS_TIME_SLIDER_HPP

#include<ramen/qwidgets/time_slider_fwd.hpp>

#include<ramen/qwidgets/time_scale.hpp>

class QSpinBox;
class QSlider;
class QDoubleSpinBox;

namespace ramen
{
namespace qwidgets
{

class RAMEN_QWIDGETS_API time_slider_t : public QWidget
{
    Q_OBJECT

public:

    time_slider_t( QWidget *parent = 0);

    void update( int start, double frame, int end);

public Q_SLOTS:

    void set_start_frame( int t);
    void set_end_frame( int t);

    void set_frame( double f);

Q_SIGNALS:

    void start_frame_changed( int t);
    void end_frame_changed( int t);
    void frame_changed( double t);

private:

    void block_all_signals( bool b);
    void adjust_frame( double frame);

    QSpinBox *start_, *end_;
    QDoubleSpinBox *current_;
    QSlider *slider_;
    time_scale_t *scale_;
};

} // qwidgets
} // ramen

#endif
