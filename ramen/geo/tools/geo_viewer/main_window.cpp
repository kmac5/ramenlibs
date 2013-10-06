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

#include<geo_viewer/main_window.hpp>

#include<ramen/config/os.hpp>

#include<ramen/gl/glew.hpp>

#include<ramen/geo/io/io.hpp>
#include<ramen/geo/io/exceptions.hpp>

#include<QApplication>
#include<QMenuBar>
#include<QMenu>
#include<QAction>
#include<QCloseEvent>
#include<QStatusBar>
#include<QDesktopWidget>
#include<QFileDialog>
#include<QMessageBox>
#include<QGLFormat>

#include<geo_viewer/scene_view.hpp>

namespace
{

QString convert_extension_list( const ramen::containers::string8_vector_t& ext_list)
{
    // TODO: get a list of extensions from ramen::geo::io.
    //static QString types( "Geo Files (*.abc) ;; Any File (*.*)");
    QString result;
    for( int i = 0; i< ext_list.size(); ++i)
    {
        result.append( "Geo Files (*.");
        result.append( ext_list[i].c_str());
        result.append( ") ;; ");
    }
    
    result.append( "Any File (*.*)");
    return result;
}

} // unnamed

main_window_t::main_window_t( QWidget *parent, Qt::WindowFlags flags) : QMainWindow( parent, flags)
{
    read_ext_list_ = convert_extension_list( ramen::geo::io::extensions_supported_for_reading());
    
    menubar_ = menuBar();

    create_grid_ = new QAction( "Grid", this);
    connect( create_grid_, SIGNAL( triggered()), this, SLOT( make_grid()));

    create_box_ = new QAction( "Box", this);
    connect( create_box_, SIGNAL( triggered()), this, SLOT( make_box()));

    create_sphere_ = new QAction( "Sphere", this);
    connect( create_sphere_, SIGNAL( triggered()), this, SLOT( make_sphere()));

    open_ = new QAction( "Open...", this);
    open_->setShortcut( QString( "Ctrl+O"));
    open_->setShortcutContext( Qt::ApplicationShortcut);
    connect( open_, SIGNAL( triggered()), this, SLOT( open()));

    quit_ = new QAction( "Quit", this);
    quit_->setShortcut( QString( "Ctrl+Q"));
    quit_->setShortcutContext( Qt::ApplicationShortcut);
    connect( quit_, SIGNAL( triggered()), this, SLOT( quit()));

    file_ = menubar_->addMenu( "File");
    new_ = file_->addMenu( "New");
        new_->addAction( create_grid_);
        new_->addAction( create_box_);
        new_->addAction( create_sphere_);
    file_->addAction( open_);
    file_->addSeparator();
    file_->addAction( quit_);

    scene_view_ = new scene_view_t( this);
    setCentralWidget( scene_view_);

    // create the status bar
    statusBar()->showMessage( "Ramen Geo Viewer");
    statusBar()->show();

    QRect screen_size = qApp->desktop()->availableGeometry();
    move( screen_size.left(), screen_size.top());
    resize( screen_size.width(), screen_size.height() - 40);
    setWindowTitle( "Ramen Geo Viewer");
}

void main_window_t::closeEvent( QCloseEvent *event)
{
    quit();
    event->accept();
}

void main_window_t::make_grid()
{
    scene_view_->make_grid_scene();
}

void main_window_t::make_box()
{
    scene_view_->make_box_scene();
}

void main_window_t::make_sphere()
{
    scene_view_->make_sphere_scene();
}

// slots
void main_window_t::open()
{
    QFileDialog dialog( this, "Open Geo...", QString::null, read_ext_list_);
    dialog.setOption( QFileDialog::DontUseNativeDialog, true);
    dialog.setFileMode( QFileDialog::ExistingFile);
    dialog.show();

    if( dialog.exec())
    {
        QStringList filenames = dialog.selectedFiles();
        std::string p = filenames[0].toStdString();
        
        try
        {
            scene_view_->load_scene( boost::filesystem::path( p));
            statusBar()->showMessage( filenames[0]);
        }
        catch( ramen::core::exception& e)
        {
            statusBar()->showMessage( "error while loading geo");
            QMessageBox::warning( this, "Error loading geo", e.what());
        }
    }
}

void main_window_t::quit()
{
    qApp->exit( 0);
}
