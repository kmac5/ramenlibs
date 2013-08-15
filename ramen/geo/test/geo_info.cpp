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

#include<iostream>

#include<boost/foreach.hpp>
#include<boost/program_options.hpp>

#include<ramen/geo/shape_vector.hpp>
#include<ramen/geo/shape_models/shape_types.hpp>
#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>
#include<ramen/geo/io/io.hpp>

namespace po = boost::program_options;

using namespace ramen;

class geo_info_visitor : public geo::const_shape_visitor
{
public:

    geo_info_visitor() {}

    virtual void visit( const geo::poly_mesh_model_t& model, const geo::shape_t& shape)
    {
        std::cout << "Shape " << shape.name() << std::endl;
        std::cout << "type = polygon mesh" << std::endl;
        print_shape_info( shape);
        print_mesh_info( model, shape);
    }

    virtual void visit( const geo::subd_mesh_model_t& model, const geo::shape_t& shape)
    {
        std::cout << "Shape " << shape.name() << std::endl;
        std::cout << "type = subd surface" << std::endl;
        print_shape_info( shape);
        print_mesh_info( model, shape);
    }

    virtual void visit( const geo::nurbs_curve_model_t& model, const geo::shape_t& shape)
    {
        std::cout << "Shape " << shape.name() << std::endl;
        std::cout << "type = nurbs curves" << std::endl;
        print_shape_info( shape);
    }

    virtual void visit( const geo::nurbs_surface_model_t& model, const geo::shape_t& shape)
    {
        std::cout << "Shape " << shape.name() << std::endl;
        std::cout << "type = nurbs surface" << std::endl;
        print_shape_info( shape);
    }

    virtual void visit( const geo::curves_model_t& model, const geo::shape_t& shape)
    {
        std::cout << "Shape " << shape.name() << std::endl;
        std::cout << "type = curves" << std::endl;
        print_shape_info( shape);
    }

    virtual void visit( const geo::points_model_t& model, const geo::shape_t& shape)
    {
        std::cout << "Shape " << shape.name() << std::endl;
        std::cout << "type = points" << std::endl;
        print_shape_info( shape);
    }

private:

    void print_shape_info( const geo::shape_t& shape)
    {
        if( shape.check_consistency())
            std::cout << "consistency check ok." << std::endl;
        else
            std::cout << "consistency check failed." << std::endl;

        std::cout << "num points "  << shape.attributes().point().const_array( geo::g_P_name).size()
                                    << std::endl;
    }

    void print_mesh_info( const geo::mesh_model_t& mesh, const geo::shape_t& shape)
    {
        std::cout << "num faces " << mesh.const_verts_per_face_array().size() << std::endl;

        if( shape.attributes().vertex().has_attribute( geo::g_uv_name))
            std::cout << "obj has uvs" << std::endl;

        if( shape.attributes().primitive().has_attribute( geo::g_material_name))
            std::cout << "obj has per-face materials" << std::endl;
    }
};

void geo_info( const char *filename)
{
    std::cout << "Reading file " << filename << " ..." << std::endl;

    geo::io::reader_t r( geo::io::reader_for_file( filename, containers::dictionary_t()));
    std::cout << "time range = [ " << r.start_time() << " ... " << r.end_time() << "]" << std::endl;

    std::cout << "shape list:" << std::endl;
    const geo::io::shape_list_t& slist = r.shape_list();
    BOOST_FOREACH( const geo::io::shape_entry_t& e, slist)
        std::cout << e.type_string() << ": " << e.name << std::endl;

    std::cout << std::endl;

    geo::shape_vector_t objects( r.read().objects());

    std::cout << "Report .........." << std::endl << std::endl;
    std::cout << "Num shapes = " << objects.size() << std::endl << std::endl;

    geo_info_visitor v;
    BOOST_FOREACH( const geo::shape_t& s, objects)
    {
        geo::apply_visitor( v, s);
        std::cout << "\n" << std::endl;
    }
}

void usage( const po::options_description& generic)
{
    std::cout << "Usage:\n";
    std::cout << "geo_info [input geo file]\n";
    std::cout << generic << "\n";
}

int main( int argc, char **argv)
{
    po::options_description generic( "Generic options");
    generic.add_options()( "help", "produce help message");

    po::options_description hidden( "Hidden options");
    hidden.add_options()( "input_file", po::value<std::string>(), "input file");

    po::options_description cmdline_options;
    cmdline_options.add( generic).add( hidden);

    po::positional_options_description p;
    p.add( "input_file", -1);

    po::variables_map vm;
    po::store( po::command_line_parser( argc, argv). options( cmdline_options).positional( p).run(), vm);
    po::notify( vm);

    if( vm.count( "help"))
    {
        usage( generic);
        return 0;
    }

    if( !vm.count( "input_file"))
    {
        std::cerr << "No geo file specified." << std::endl;
        usage( generic);
        return 1;
    }

    try
    {
        geo_info( vm["input_file"].as<std::string>().c_str());
    }
    catch( core::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    catch( std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    catch( ...)
    {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}
