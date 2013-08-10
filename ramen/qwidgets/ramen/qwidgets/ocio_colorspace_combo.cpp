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

#include<ramen/qwidgets/ocio_colorspace_combo.hpp>

#include<cassert>

#include<OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;

namespace ramen
{
namespace qwidgets
{

ocio_colorspace_combo_t::ocio_colorspace_combo_t( QWidget *parent) : ocio_combo_t( parent)
{
    OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();

    for( int i = 0, e = config->getNumColorSpaces(); i < e; ++i)
        addItem( config->getColorSpaceNameByIndex( i));

    set_default();
    connect( this, SIGNAL( currentIndexChanged( int)), this, SLOT( combo_index_changed( int)));
}

QString ocio_colorspace_combo_t::get_current_colorspace() const
{
    return currentText();
}

void ocio_colorspace_combo_t::set_colorspace( const QString& cs)
{
    int index = index_for_string( cs);
    assert( index != -1);

    setCurrentIndex( index);
}

bool ocio_colorspace_combo_t::set_colorspace_or_default( const QString& cs)
{
    int index = index_for_string( cs);

    if( index != -1)
    {
        setCurrentIndex( index);
        return true;
    }
    else
    {
        set_default();
        return false;
    }
}

void ocio_colorspace_combo_t::set_default()
{
    OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();
    QString default_cs_name = config->getColorSpace( OCIO::ROLE_SCENE_LINEAR)->getName();

    int index = 0;

    for(int i = 0, e = config->getNumColorSpaces(); i < e; ++i)
    {
        if( config->getColorSpaceNameByIndex( i) == default_cs_name)
            index = i;
    }

    setCurrentIndex( index);
}

void ocio_colorspace_combo_t::combo_index_changed( int indx)
{
    colorspace_changed( currentText());
}

} // qwidgets
} // ramen
