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

#ifndef RAMEN_QWIDGETS_OCIO_VIEW_COMBO_HPP
#define RAMEN_QWIDGETS_OCIO_VIEW_COMBO_HPP

#include<ramen/qwidgets/ocio_combo.hpp>

namespace ramen
{
namespace qwidgets
{

class ocio_view_combo_t : public ocio_combo_t
{
    Q_OBJECT

public:

    ocio_view_combo_t( QWidget *parent = 0);

    QString get_current_view() const;

public Q_SLOTS:

    void set_view( const QString& s);

    void update_views( const QString& display);

Q_SIGNALS:

    void view_changed( const QString&);

private Q_SLOTS:

    void combo_index_changed( int indx);
};

} // qwidgets
} // ramen

#endif
