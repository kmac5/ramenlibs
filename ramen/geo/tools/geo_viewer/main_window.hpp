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

#ifndef GEO_VIEWER_MAIN_WINDOW_HPP
#define GEO_VIEWER_MAIN_WINDOW_HPP

#include<QMainWindow>

#include<geo_viewer/scene_view_fwd.hpp>

class QAction;
class QActionGroup;
class QMenu;
class QMenuBar;
class QToolBar;

class main_window_t : public QMainWindow
{
    Q_OBJECT

public:

    main_window_t( QWidget *parent = 0, Qt::WindowFlags flags = 0);

protected:

    void closeEvent( QCloseEvent *event);

public Q_SLOTS:

    void make_grid();
    void make_box();
    void make_sphere();
    
    void open();
    void quit();

private:

    QMenuBar *menubar_;

    QMenu *file_;
        QMenu *new_;
            QAction *create_grid_;
            QAction *create_box_;
            QAction *create_sphere_;
        QAction *open_;
        QAction *quit_;

    scene_view_t *scene_view_;
    
    QString read_ext_list_;    
};

#endif
