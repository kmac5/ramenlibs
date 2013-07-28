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

#ifndef RAMEN_GEO_IO_ALEMBIC_READER_MODEL_HPP
#define RAMEN_GEO_IO_ALEMBIC_READER_MODEL_HPP

#include<ramen/geo/io/reader_interface.hpp>

#include<boost/static_assert.hpp>

#include<Alembic/AbcGeom/All.h>
#include<Alembic/AbcCoreHDF5/All.h>

#include<ramen/math/transform44_stack.hpp>

#include<ramen/geo/shape_fwd.hpp>
#include<ramen/geo/shape_models/mesh_model_fwd.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

class alembic_reader_model_t : public reader_interface_t
{
public:

    alembic_reader_model_t( const char *filename, const core::dictionary_t& options);

    virtual void release();

    virtual const scene_t read( const shape_list_filter_t& filter);

private:

    void get_archive_info( Alembic::AbcGeom::IObject obj);

    void do_get_archive_info( Alembic::AbcGeom::IObject obj, const core::string8_t& name_prefix);

    template<class Schema>
    void get_time_range( const Schema& schema);

    void read_objects( Alembic::AbcGeom::IObject obj,
                        const core::string8_t& name_prefix,
                        const shape_list_filter_t& filter,
                        shape_vector_t& objects);

    bool read_xform( Alembic::AbcGeom::IXform xform);
    void apply_xform( shape_t& shape);

    void read_points( Alembic::AbcGeom::IPoints obj,
                      const core::string8_t& name,
                      shape_vector_t& objects);

    void read_poly_mesh( Alembic::AbcGeom::IPolyMesh obj,
                         const core::string8_t& name,
                         shape_vector_t& objects);

    void read_subd( Alembic::AbcGeom::ISubD obj,
                    const core::string8_t& name,
                    shape_vector_t& objects);

    void read_nupatch( Alembic::AbcGeom::INuPatch obj,
                       const core::string8_t& name,
                       shape_vector_t& objects);

    void read_curves( Alembic::AbcGeom::ICurves obj,
                      const core::string8_t& name,
                      shape_vector_t& objects);

    void read_mesh( shape_t& shape,
                    mesh_model_t& mesh,
                    Alembic::Abc::P3fArraySamplePtr positions,
                    Alembic::Abc::Int32ArraySamplePtr counts,
                    Alembic::Abc::Int32ArraySamplePtr indices,
                    Alembic::Abc::V3fArraySamplePtr velocities);

    void read_arbitrary_geo_params( Alembic::Abc::ICompoundProperty params, shape_t& shape);

    template<class GeoParam>
    void read_geo_param( Alembic::Abc::ICompoundProperty parent,
                         const Alembic::AbcGeom::PropertyHeader& prop_header,
                         shape_t& shape);

    boost::uint32_t abc_type_to_shape_type( const Alembic::AbcGeom::MetaData& md) const;

    // options
    float time_;
    bool use_paths_as_names_;
    bool apply_transforms_;
    bool read_arbitrary_;

    math::transform44_stack_t<double> xform_stack_;
};

} // io
} // geo
} // ramen

#endif
