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

#include<ramen/qwidgets/ocio_combo.hpp>

#include<cassert>
#include<vector>
#include<algorithm>

namespace ramen
{
namespace qwidgets
{
namespace
{

std::vector<ocio_combo_t*> g_ocio_combos_;

} // unnamed

ocio_combo_t::ocio_combo_t( QWidget *parent) : QComboBox( parent)
{
    g_ocio_combos_.push_back( this);
}

ocio_combo_t::~ocio_combo_t()
{
    g_ocio_combos_.erase( std::remove( g_ocio_combos_.begin(), g_ocio_combos_.end(), this), g_ocio_combos_.end());
}

void ocio_combo_t::ocio_config_changed()
{
    // TODO: implement this...
    assert( false);

    // for each ocio_combo_t created
    // update combo
}

int ocio_combo_t::index_for_string( const QString& s) const
{
    return findText( s);
}

} // qwidgets
} // ramen
