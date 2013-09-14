// Copyright (c) 2010 Esteban Tovagliari
// Licensed under the terms of the CDDL License.
// See CDDL_LICENSE.txt for a copy of the license.

#include<ramen/qwidgets/time_slider.hpp>

#include<QHBoxLayout>
#include<QSpinBox>
#include<QDoubleSpinBox>

namespace ramen
{
namespace qwidgets
{

time_slider_t::time_slider_t( QWidget *parent) : QWidget( parent)
{
    start_ = new QSpinBox();
    start_->setRange( -32768, 32768);
    start_->setValue( 1);
    start_->setSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed);

    end_ = new QSpinBox();
    end_->setRange( -32768, 32768);
    end_->setValue( 100);
    end_->setSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed);

    current_ = new QDoubleSpinBox();
    current_->setRange(1, 100);
    current_->setValue( 1);
    current_->setDecimals( 0);
    current_->setSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed);

    scale_ = new time_scale_t();
    scale_->setRange(1, 100);
    scale_->setValue( 1);

    connect( start_	 , SIGNAL( valueChanged( int)), this, SLOT( set_start_frame( int)));
    connect( end_	 , SIGNAL( valueChanged( int)), this, SLOT( set_end_frame( int)));
    connect( scale_	 , SIGNAL( valueChanged( double)), this, SLOT( set_frame( double)));
    connect( current_, SIGNAL( valueChanged( double)), this, SLOT( set_frame( double)));

    QHBoxLayout *layout = new QHBoxLayout;
    layout->addWidget( start_);
    layout->addWidget( current_);
    layout->addWidget( scale_);
    layout->addWidget( end_);
    setLayout(layout);

    setSizePolicy( QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void time_slider_t::update_state( int start, double frame, int end)
{
    block_all_signals( true);

    start_->setValue( start);
    end_->setValue( end);

    current_->setMinimum( start);
    current_->setMaximum( end);
    current_->setValue( frame);

    scale_->setMinimum( start);
    scale_->setMaximum( end);
    scale_->setValue( frame);

    block_all_signals( false);
}

void time_slider_t::set_start_frame( int t)
{
    block_all_signals( true);

    double cur_frame = current_->value();
    int new_start = std::min( t, end_->value());
    start_->setValue( new_start);

    current_->setMinimum( start_->value());
    scale_->setMinimum( start_->value());

    block_all_signals( false);

    start_frame_changed( start_->value());
    adjust_frame( cur_frame);
}

void time_slider_t::set_end_frame( int t)
{
    block_all_signals( true);

    double cur_frame = current_->value();
    int new_end = std::max( t, start_->value());
    end_->setValue( new_end);

    current_->setMaximum( end_->value());
    scale_->setMaximum( end_->value());

    block_all_signals( false);

    end_frame_changed( end_->value());
    adjust_frame( cur_frame);
}

void time_slider_t::set_frame( double t)
{
    block_all_signals( true);
    scale_->setValue( t);
    current_->setValue( t);
    block_all_signals( false);
    frame_changed( t);
}

void time_slider_t::block_all_signals( bool b)
{
    start_->blockSignals( b);
    end_->blockSignals( b);
    current_->blockSignals( b);
    scale_->blockSignals( b);
}

void time_slider_t::adjust_frame( double frame)
{
    double new_value = frame;
    if( new_value < start_->value())
        new_value = start_->value();

    if( new_value > end_->value())
        new_value = end_->value();

    if( new_value != frame)
    {
        block_all_signals( true);
        current_->setValue( new_value);
        scale_->setValue( new_value);
        block_all_signals( false);
        frame_changed( new_value);
    }
}

} // qwidgets
} // ramen
