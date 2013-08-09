// Copyright (c) 2010 Esteban Tovagliari
// Licensed under the terms of the CDDL License.
// See CDDL_LICENSE.txt for a copy of the license.

#include<ramen/qwidgets/time_scale.hpp>

#include<algorithm>

#include<QHBoxLayout>
#include<QPainter>
#include<QMouseEvent>
#include<QPaintEvent>

#include<ramen/math/cmath.hpp>

#include<ramen/iterators/nice_numbers.hpp>

namespace ramen
{
namespace qwidgets
{
namespace
{

template<class T>
T clamp( T x, T lo, T hi)
{
    if( x < lo)
        return lo;

    if( x > hi)
        return hi;

    return x;
}

} // unnamed

time_scale_t::time_scale_t( QWidget *parent) : QWidget( parent)
{
    min_value_ = 1;
    max_value_ = 100;
    value_ = 1;
    drag_ = false;
}

void time_scale_t::setRange( int lo, int hi)
{
    min_value_ = lo;
    max_value_ = hi;
    update();
}

void time_scale_t::setMinimum( int m)
{
    min_value_ = m;
    update();
}

void time_scale_t::setMaximum( int m)
{
    max_value_ = m;
    update();
}

void time_scale_t::setValue( double v)
{
    double new_val = clamp( v, (double) min_value_, (double) max_value_);

    if( value_ != new_val)
    {
        value_ = new_val;
        valueChanged( v);
        update();
    }
}

int time_scale_t::round_halfup( float x) const
{
    int result = math::cmath<double>::floor( math::cmath<double>::fabs( x) + 0.5);
    return ( x < 0.0) ? -result : result;
}

int time_scale_t::frame_from_mouse_pos( int x) const
{
    float f = ( float) x / width();
    f = f * ( max_value_ - min_value_) + min_value_;
    return clamp( round_halfup( f), min_value_, max_value_);
}

void time_scale_t::mousePressEvent( QMouseEvent *event)
{
    if( event->button() == Qt::LeftButton)
    {
        drag_ = true;
        last_x_ = event->x();
        setValue( frame_from_mouse_pos( last_x_));
    }

    event->accept();
}

void time_scale_t::mouseMoveEvent( QMouseEvent *event)
{
    if( drag_)
    {
        if( last_x_ != event->x())
            setValue( frame_from_mouse_pos( event->x()));

        last_x_ = event->x();
    }

    event->accept();
}

void time_scale_t::mouseReleaseEvent( QMouseEvent *event)
{
    event->accept();
}

void time_scale_t::paintEvent ( QPaintEvent *event)
{
    QPainter painter( this);
    painter.setRenderHint( QPainter::Antialiasing);

    QPen pen;
    pen.setColor( QColor( 0, 0, 0));
    pen.setWidth( 1);
    painter.setPen( pen);

    painter.drawLine( 0, 7, width(), 7);

    const int spacing = 50;
    int nticks = math::cmath<double>::floor( (double) width() / spacing);
    for( iterators::nice_numbers_t it( min_value_, max_value_, nticks), e; it != e; ++it)
    {
        float x = *it;
        float sx = ( x - min_value_) / ( max_value_ - min_value_) * width();
        painter.drawLine( QPointF( sx, 2), QPointF( sx, 12));
        painter.drawText( QPoint( sx, height()), QString::number( x));
    }

    pen.setColor( QColor( 255, 0, 0));
    pen.setWidth( 3);
    painter.setPen( pen);

    float x = (float) (value_ - min_value_) / (float) ( max_value_ - min_value_) * width();
    painter.drawLine( QPointF( x, 0), QPointF( x, height()));

    event->accept();
}

} // qwidgets
} // ramen
