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

#include<ramen/qwidgets/ocio_view_combo.hpp>

#include<cassert>

#include<OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

namespace ramen
{
namespace qwidgets
{

ocio_view_combo_t::ocio_view_combo_t( QWidget *parent) : ocio_combo_t( parent)
{
    OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();
    int num_views = config->getNumViews( config->getDefaultDisplay());
    QString default_view = config->getDefaultView( config->getDefaultDisplay());
    int default_index = 0;

    for( int i = 0; i < num_views; ++i)
    {
        QString view_name = config->getView( config->getDefaultDisplay(), i);

        if( view_name == default_view)
            default_index = i;

        addItem( view_name);
    }

    setCurrentIndex( default_index);
    connect( this, SIGNAL( currentIndexChanged( int)), this, SLOT( combo_index_changed( int)));
}

QString ocio_view_combo_t::get_current_view() const
{
    return currentText();
}

void ocio_view_combo_t::set_view( const QString& v)
{
    int index = index_for_string( v);
    assert( index != -1);

    setCurrentIndex( index);
}

void ocio_view_combo_t::update_views( const QString& display)
{
    blockSignals( true);

    QString old_view = get_current_view();
    clear();

    QByteArray raw_display_name = display.toAscii();
    OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();
    int num_views = config->getNumViews( raw_display_name.data());
    QString default_view = config->getDefaultView( raw_display_name.data());
    int default_index = 0;
    int new_index = -1;

    for( int i = 0; i < num_views; ++i)
    {
        QString view_name = config->getView( raw_display_name.data(), i);

        if( view_name == default_view)
            default_index = i;

        if( view_name == old_view)
            new_index = i;

        addItem( view_name);
    }

    blockSignals( false);

    if( new_index == -1)
        new_index = default_index;

    setCurrentIndex( new_index);
}

void ocio_view_combo_t::combo_index_changed( int indx)
{
    view_changed( currentText());
}

} // qwidgets
} // ramen
