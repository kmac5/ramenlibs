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

#include<ramen/geo/io/alembic/alembic_reader_model.hpp>

#include<boost/static_assert.hpp>

#include<ramen/geo/shape_vector.hpp>

#include<ramen/geo/shape_models/mesh_model.hpp>
#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>

#include<ramen/geo/visitors/transform.hpp>
#include<ramen/geo/visitors/shape_to_points.hpp>

using namespace Alembic;

namespace ramen
{
namespace geo
{
namespace io
{

namespace
{
template<class VecType, class ResultType>
struct convert
{
    ResultType operator()( const VecType& p) const
    {
        return ResultType( p.x, p.y, p.z);
    }
};

} // unnamed

alembic_reader_model_t::alembic_reader_model_t( const char *filename,
                                                const containers::dictionary_t& options) : reader_interface_t( filename, options)
{
    use_paths_as_names_ = containers::get_optional( options, core::name_t( "paths_as_names"), false);

    start_time_ = std::numeric_limits<double>::max();
    end_time_ = -start_time_;

    AbcGeom::IArchive archive( AbcCoreHDF5::ReadArchive(), filename);
    get_archive_info( archive.getTop());

    time_ = get_optional( options, g_time_name, start_time_);
    apply_transforms_ = containers::get_optional( options, core::name_t( "apply_transforms"), true);
    read_arbitrary_   = containers::get_optional( options, core::name_t( "read_arbitrary"), true);
}

void alembic_reader_model_t::release() { delete this;}

void alembic_reader_model_t::get_archive_info( AbcGeom::IObject obj)
{
    // skip the root node from most processing...

    // build shape list
    core::string8_t root( "/");

    for( int i = 0, e = obj.getNumChildren() ; i < e ; ++i)
        do_get_archive_info( AbcGeom::IObject( obj, obj.getChildHeader( i).getName()), root);

    // if the file does not have anim, set time range to zero.
    if( end_time_ < start_time_)
        start_time_ = end_time_ = 0.0f;
}

void alembic_reader_model_t::do_get_archive_info( AbcGeom::IObject obj,
                                                  const core::string8_t& name_prefix)
{
    const AbcGeom::MetaData& md = obj.getMetaData();

    // shape list
    core::string8_t name;

    if( use_paths_as_names_)
        name = boost::move( core::make_string( name_prefix.c_str(), obj.getName().c_str()));
    else
        name = obj.getName().c_str();

    if( uint32_t shape_type = abc_type_to_shape_type( md) != 0)
        shape_list_.emplace_back( name, shape_type);

    // time range
    if( AbcGeom::IXform::matches( md))
        get_time_range( AbcGeom::IXform( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::ICameraSchema::matches( md))
        get_time_range( AbcGeom::ICamera( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::ILightSchema::matches( md))
        get_time_range( AbcGeom::ILight( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::IPolyMeshSchema::matches( md))
        get_time_range( AbcGeom::IPolyMesh( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::ISubDSchema::matches( md))
        get_time_range( AbcGeom::ISubD( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::INuPatchSchema::matches( md))
        get_time_range( AbcGeom::INuPatch( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::ICurvesSchema::matches( md))
        get_time_range( AbcGeom::ICurves( obj, AbcGeom::kWrapExisting).getSchema());
    else if( AbcGeom::IPointsSchema::matches( md))
        get_time_range( AbcGeom::IPoints( obj, AbcGeom::kWrapExisting).getSchema());

    // recurse
    if( use_paths_as_names_)
        name.push_back( '/');

    for( int i = 0, e = obj.getNumChildren() ; i < e ; ++i)
        do_get_archive_info( AbcGeom::IObject( obj, obj.getChildHeader( i).getName()), name);
}

template<class Schema>
void alembic_reader_model_t::get_time_range( const Schema& schema)
{
    Abc::TimeSamplingPtr tsmp = schema.getTimeSampling();

    if( !schema.isConstant())
    {
        int num_samps = schema.getNumSamples();

        if( num_samps > 0)
        {
            start_time_  = std::min( start_time_, static_cast<double>( tsmp->getSampleTime( 0)));
            end_time_    = std::max( end_time_  , static_cast<double>( tsmp->getSampleTime( num_samps - 1)));
        }
    }
}

const scene_t alembic_reader_model_t::read( const shape_list_filter_t& filter)
{
    AbcGeom::IArchive archive( AbcCoreHDF5::ReadArchive(), filename().c_str());
    AbcGeom::IObject top( archive.getTop());

    core::string8_t root( "/");
    scene_t scene;

    xform_stack_.reset();

    for( int i = 0, e = top.getNumChildren() ; i < e ; ++i)
        read_objects( AbcGeom::IObject( top, top.getChildHeader( i).getName()),
                      root, filter, scene.objects());

    return scene;
}

void alembic_reader_model_t::read_objects( AbcGeom::IObject obj,
                                            const core::string8_t& name_prefix,
                                            const shape_list_filter_t& filter,
                                            shape_vector_t& objects)
{
    const AbcGeom::MetaData& md = obj.getMetaData();
    core::string8_t name;

    if( use_paths_as_names_)
        name = core::make_string( name_prefix.c_str(), obj.getName().c_str());
    else
        name = obj.getName().c_str();

    bool pop_transform = false;
    if( AbcGeom::IXformSchema::matches( md))
        pop_transform = read_xform( AbcGeom::IXform( obj, AbcGeom::kWrapExisting));
    else
    {
        if( uint32_t shape_type = abc_type_to_shape_type( md) != 0)
        {
            if( filter( name, shape_type))
            {
                if( AbcGeom::IPointsSchema::matches( md))
                    read_points( AbcGeom::IPoints( obj, AbcGeom::kWrapExisting), name, objects);
                else if( AbcGeom::IPolyMeshSchema::matches( md))
                    read_poly_mesh( AbcGeom::IPolyMesh( obj, AbcGeom::kWrapExisting), name, objects);
                else if( AbcGeom::ISubDSchema::matches( md))
                    read_subd( AbcGeom::ISubD( obj, AbcGeom::kWrapExisting), name, objects);
                else if( AbcGeom::INuPatchSchema::matches( md))
                    read_nupatch( AbcGeom::INuPatch( obj, AbcGeom::kWrapExisting), name, objects);
                else if( AbcGeom::ICurvesSchema::matches( md))
                    read_curves( AbcGeom::ICurves( obj, AbcGeom::kWrapExisting), name, objects);
                else
                    throw core::not_implemented();
            }
        }
    }

    // recurse
    if( use_paths_as_names_)
        name.push_back( '/');

    for( int i = 0, e = obj.getNumChildren() ; i < e ; ++i)
        read_objects( AbcGeom::IObject( obj, obj.getChildHeader( i).getName()), name, filter, objects);

    if( pop_transform)
        xform_stack_.pop();
}

bool alembic_reader_model_t::read_xform( AbcGeom::IXform xform)
{
    if( !apply_transforms_)
        return false;

    if( !xform)
        return false;

    const AbcGeom::IXformSchema& xform_schema( xform.getSchema());

    if( !xform_schema)
        return false;

    if( xform_schema.isConstantIdentity())
        return false;

    AbcGeom::XformSample sample( xform_schema.getValue( Abc::ISampleSelector( time_)));
    Imath::M44d matrix( sample.getMatrix());
    math::matrix44d_t m( &matrix[0][0]);
    math::transform44d_t x( m);
    xform_stack_.push();

    if( sample.getInheritsXforms())
        xform_stack_.concat_transform( x);
    else
        xform_stack_.set_transform( x);

    return true;
}

void alembic_reader_model_t::apply_xform( shape_t& shape)
{
    if( apply_transforms_)
    {
        transform_visitor<double> v( xform_stack_.top());
        apply_visitor( v, shape);
    }
}

void alembic_reader_model_t::read_points( AbcGeom::IPoints obj,
                                          const core::string8_t& name,
                                          shape_vector_t& objects)
{
    if( !obj)
        return;

    const AbcGeom::IPointsSchema& schema( obj.getSchema());

    if( !schema)
        return;

    AbcGeom::IPointsSchema::Sample sample;
    schema.get( sample, Abc::ISampleSelector( time_));

    if( !sample)
        return;

    std::size_t num_points = 0;
    shape_t shape( shape_t::create_points());

    // P.
    if( sample.getPositions())
    {
        num_points = sample.getPositions()->size();

        if( num_points == 0)
            return;

        shape.attributes().point().insert( g_P_name, core::point3f_k);
        arrays::array_ref_t<math::point3f_t> Pref( shape.attributes().point().array( g_P_name));
        Pref.reserve( num_points);

        std::transform( sample.getPositions()->get(),
                        sample.getPositions()->get() + num_points,
                        std::back_inserter( Pref),
                        convert<Imath::V3f, math::point3f_t>());
    }
    else
    {
        // if there are no points, there's no point...
        // TODO: handle the errors here...
        return;
    }

    // Ids.
    if( sample.getIds())
    {
        shape.attributes().point().insert( g_id_name, core::uint32_k);
        arrays::array_ref_t<boost::uint32_t> Iref( shape.attributes().point().array( g_id_name));
        Iref.reserve( num_points);
        const uint64_t *src = sample.getIds()->get();
        std::copy( src, src + num_points, std::back_inserter( Iref));
    }

    // Velocity
    if( sample.getVelocities())
    {
        shape.attributes().point().insert( g_velocity_name, core::vector3f_k);
        arrays::array_ref_t<math::vector3f_t> Vref( shape.attributes().point().array( g_velocity_name));
        Vref.reserve( num_points);
        std::transform( sample.getVelocities()->get(),
                        sample.getVelocities()->get() + num_points,
                        std::back_inserter( Vref),
                        convert<Imath::V3f, math::vector3f_t>());
    }

    /*
    // width
    if( AbcGeom::IFloatGeomParam width_param = schema.getWidthsParam())
    {
        core::name_t width_name( "width");

        switch( width_param.getScope())
        {
            case AbcGeom::kVaryingScope:
            case AbcGeom::kVertexScope:
            case AbcGeom::kFacevaryingScope:
            {
                shape.attributes().point().insert( width_name, arrays::float_k);
                arrays::array_ref_t<float> Wref( shape.attributes().point().array( width_name));
                Wref.reserve( num_points);

                AbcGeom::IFloatGeomParam::Sample width_sample;
                width_param.getExpanded( width_sample, Abc::ISampleSelector( time_));
                const float *src = width_sample.getVals()->get();
                std::copy( src, src + num_points, std::back_inserter( Wref));
            }
            break;

            case AbcGeom::kUniformScope:
            case AbcGeom::kConstantScope:
            case AbcGeom::kUnknownScope:
            {
                AbcGeom::IFloatGeomParam::Sample width_sample;
                width_param.getExpanded( width_sample, Abc::ISampleSelector( time_));
                const float *src = width_sample.getVals()->get();
                shape.attributes().constant()[width_name] = core::poly_regular_t( *src);
            }
            break;

            default:
            return;
        }
    }

    if( read_arbitrary_)
        read_arbitrary_geo_params( schema.getArbGeomParams(), shape);
    */

    apply_xform( shape);
    shape.set_name( name);
    objects.push_back( boost::move( shape));
}

void alembic_reader_model_t::read_poly_mesh( AbcGeom::IPolyMesh obj,
                                             const core::string8_t& name,
                                             shape_vector_t& objects)
{
    if( !obj)
        return;

    const AbcGeom::IPolyMeshSchema& schema( obj.getSchema());

    if( !schema)
        return;

    AbcGeom::IPolyMeshSchema::Sample sample;
    schema.get( sample, Abc::ISampleSelector( time_));

    if( !sample)
        return;

    if( std::size_t num_points = sample.getPositions()->size() == 0)
        return;

    shape_t shape( shape_t::create_poly_mesh());

    read_mesh( shape,
               shape_cast<poly_mesh_model_t>( shape),
               sample.getPositions(),
               sample.getFaceCounts(),
               sample.getFaceIndices(),
               sample.getVelocities());

    if( read_arbitrary_)
        read_arbitrary_geo_params( schema.getArbGeomParams(), shape);

    apply_xform( shape);
    shape.set_name( name);
    objects.push_back( boost::move( shape));
}

void alembic_reader_model_t::read_subd( AbcGeom::ISubD obj,
                                        const core::string8_t& name,
                                        shape_vector_t& objects)
{
    if( !obj)
        return;

    const AbcGeom::ISubDSchema& schema( obj.getSchema());

    if( !schema)
        return;

    AbcGeom::ISubDSchema::Sample sample;
    schema.get( sample, Abc::ISampleSelector( time_));

    if( !sample)
        return;

    if( std::size_t num_points = sample.getPositions()->size() == 0)
        return;

    shape_t shape( shape_t::create_subd_mesh());

    read_mesh( shape,
               shape_cast<subd_mesh_model_t>( shape),
               sample.getPositions(),
               sample.getFaceCounts(),
               sample.getFaceIndices(),
               sample.getVelocities());

    // SubDs specific
    //shape.attributes().constant()[ name_t( "subd_scheme")] = core::string8_t( sample.getSubdivisionScheme().c_str());

    if( read_arbitrary_)
        read_arbitrary_geo_params( schema.getArbGeomParams(), shape);

    apply_xform( shape);
    shape.set_name( name);
    objects.push_back( boost::move( shape));
}

void alembic_reader_model_t::read_nupatch( AbcGeom::INuPatch obj,
                                           const core::string8_t& name,
                                           shape_vector_t& objects)
{
    // TODO: implement this.
}

void alembic_reader_model_t::read_curves( AbcGeom::ICurves obj,
                                          const core::string8_t& name,
                                          shape_vector_t& objects)
{
    //AbcGeom::ICurvesSchema::Sample sample;
}

void alembic_reader_model_t::read_mesh( shape_t& shape,
                                        mesh_model_t& mesh,
                                        Abc::P3fArraySamplePtr positions,
                                        Abc::Int32ArraySamplePtr counts,
                                        Abc::Int32ArraySamplePtr indices,
                                        Abc::V3fArraySamplePtr velocities)
{
    std::size_t num_points = positions->size();

    // P
    {
        shape.attributes().point().insert( g_P_name, core::point3f_k);
        arrays::array_ref_t<math::point3f_t> Pref( shape.attributes().point().array( g_P_name));
        Pref.reserve( num_points);
        std::transform( positions->get(),
                        positions->get() + num_points,
                        std::back_inserter( Pref),
                        convert<Imath::V3f, math::point3f_t>());
    }

    // face vertex counts
    {
        arrays::array_ref_t<boost::uint32_t> ref( mesh.verts_per_face_array());
        ref.reserve( counts->size());
        std::copy( counts->get(),
                   counts->get() + counts->size(),
                   std::back_inserter( ref));
    }

    // face vertex indices
    {
        arrays::array_ref_t<boost::uint32_t> ref( mesh.face_vert_indices_array());
        ref.reserve( indices->size());
        std::copy( indices->get(),
                   indices->get() + indices->size(),
                   std::back_inserter( ref));
    }

    // velocity
    if( velocities)
    {
        shape.attributes().point().insert( g_velocity_name, core::vector3f_k);
        arrays::array_ref_t<math::vector3f_t> Vref( shape.attributes().point().array( g_velocity_name));
        Vref.reserve( num_points);
        std::transform( velocities->get(),
                        velocities->get() + num_points,
                        std::back_inserter( Vref),
                        convert<Imath::V3f, math::vector3f_t>());

    }

    // if the mesh has no faces, convert it to points.
    if( counts->size() == 0)
    {
        shape_to_points_visitor v;
        apply_visitor( v, shape);
    }
}

void alembic_reader_model_t::read_arbitrary_geo_params( Abc::ICompoundProperty params,
                                                        shape_t& shape)
{
    if( !params.valid())
        return;

    for( int i = 0, e = params.getNumProperties(); i < e; ++i)
    {
        const Abc::PropertyHeader& prop_header = params.getPropertyHeader( i);

        if( AbcGeom::IFloatGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IFloatGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IDoubleGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IDoubleGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IInt32GeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IInt32GeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IStringGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IStringGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IV2fGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IV2fGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IV3fGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IV3fGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IN3fGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IN3fGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IC3fGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IC3fGeomParam>( params, prop_header, shape);
        }
        else if( AbcGeom::IP3fGeomParam::matches( prop_header))
        {
            read_geo_param<AbcGeom::IP3fGeomParam>( params, prop_header, shape);
        }
        else
        {
            // TODO: Add some kind of warning here...
        }
    }
}

template<class GeoParam>
void alembic_reader_model_t::read_geo_param( Abc::ICompoundProperty parent,
                                             const AbcGeom::PropertyHeader& prop_header,
                                             shape_t& shape)
{
    GeoParam param( parent, prop_header.getName());
    core::name_t param_name( prop_header.getName().c_str());

    Abc::ISampleSelector sample_selector( time_);
    typename GeoParam::sample_type param_sample;
    param.getExpanded( param_sample, sample_selector);

    switch( param_sample.getScope())
    {
        case AbcGeom::kVaryingScope:
        case AbcGeom::kVertexScope:
            // point attr;
        break;

        case AbcGeom::kFacevaryingScope:
            // vertex attr;
        break;

        case AbcGeom::kUniformScope:
            // face attr
        break;

        case AbcGeom::kConstantScope:
        case AbcGeom::kUnknownScope:
            // constant attr
        break;
    }
}

uint32_t alembic_reader_model_t::abc_type_to_shape_type( const AbcGeom::MetaData& md) const
{
    if( AbcGeom::IPolyMeshSchema::matches( md))
        return poly_mesh_shape_k;
    else if( AbcGeom::ISubDSchema::matches( md))
        return subd_mesh_shape_k;
    else if( AbcGeom::INuPatchSchema::matches( md))
        return nurbs_surface_shape_k;
    else if( AbcGeom::IPointsSchema::matches( md))
        return points_shape_k;
    else if( AbcGeom::INuPatchSchema::matches( md))
        return curves_shape_k;

    return 0;
}

} // io
} // geo
} // ramen
