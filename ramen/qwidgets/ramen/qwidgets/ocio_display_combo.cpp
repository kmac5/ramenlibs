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

#include<ramen/qwidgets/ocio_display_combo.hpp>

#include<cassert>

#include<OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

namespace ramen
{
namespace qwidgets
{

ocio_display_combo_t::ocio_display_combo_t( QWidget *parent) : ocio_combo_t( parent)
{
    OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();
    QString default_display = config->getDefaultDisplay();
    int num_device_names = config->getNumDisplays();
    int default_index = 0;

    for( int i = 0; i < num_device_names; ++i)
    {
        QString disp_name = config->getDisplay( i);
        addItem( disp_name);

        if( disp_name == default_display)
            default_index = i;
    }

    setCurrentIndex( default_index);
    connect( this, SIGNAL( currentIndexChanged(int)), this, SLOT( combo_index_changed(int)));
}

void ocio_display_combo_t::set_display( const QString& cs)
{
    int index = index_for_string( cs);
    assert( index != -1);

    setCurrentIndex( index);
}

void ocio_display_combo_t::set_default()
{
    OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();
    QString default_display = config->getDefaultDisplay();
    int num_device_names = config->getNumDisplays();
    int default_index = 0;

    for( int i = 0; i < num_device_names; ++i)
    {
        QString disp_name = config->getDisplay( i);

        if( disp_name == default_display)
        {
            default_index = i;
            break;
        }
    }

    setCurrentIndex( default_index);
}

void ocio_display_combo_t::combo_index_changed( int indx)
{
    display_changed( currentText());
}

} // qwidgets
} // ramen
