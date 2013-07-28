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

#include<ramen/geo/shape.hpp>

#include<algorithm>

#include<ramen/core/memory.hpp>

#include<ramen/geo/Concepts/ShapeModelConcept.hpp>
#include<ramen/geo/global_names.hpp>

#include<ramen/geo/shape_models/poly_mesh_model.hpp>
#include<ramen/geo/shape_models/subd_mesh_model.hpp>
#include<ramen/geo/shape_models/nurbs_curve_model.hpp>
#include<ramen/geo/shape_models/nurbs_surface_model.hpp>
#include<ramen/geo/shape_models/curves_model.hpp>
#include<ramen/geo/shape_models/points_model.hpp>

namespace ramen
{
namespace geo
{

struct shape_t::shape_interface_t
{
    virtual ~shape_interface_t() {}

    virtual shape_interface_t *copy() const = 0;

    virtual shape_type_t type() const = 0;

    virtual void accept( const_shape_visitor& v, const shape_t& shape) const = 0;
    virtual void accept( shape_visitor& v, shape_t& shape) = 0;

    virtual const void *get_model() const = 0;
    virtual void *get_model() = 0;

    virtual bool check_consistency( const shape_attributes_t& attrs) const = 0;
};

template<class T>
class shape_model_t : public shape_t::shape_interface_t
{
public:

    explicit shape_model_t( const T& x) : impl_( x) {}

    virtual shape_model_t<T> *copy() const
    {
        return new shape_model_t( *this);
    }

    virtual shape_type_t type() const
    {
        return shape_model_traits<T>::type();
    }

    virtual void accept( const_shape_visitor& v, const shape_t& shape) const
    {
        v.visit( impl_, shape);
    }

    virtual void accept( shape_visitor& v, shape_t& shape)
    {
        v.visit( impl_, shape);
    }

    virtual const void *get_model() const
    {
        return reinterpret_cast<const void*>( &impl_);
    }

    virtual void *get_model()
    {
        return reinterpret_cast<void*>( &impl_);
    }

    virtual bool check_consistency( const shape_attributes_t& attrs) const
    {
        return impl_.check_consistency( attrs);
    }

private:

    BOOST_CONCEPT_ASSERT(( ShapeModelConcept<T>));

    T impl_;
};

shape_t shape_t::create_poly_mesh()
{
    typedef shape_model_t<poly_mesh_model_t> model_type;
    typedef core::auto_ptr_t<model_type> auto_ptr_type;

    auto_ptr_type tmp( new model_type( poly_mesh_model_t()));

    shape_t shape;
    shape.attributes().point().insert( g_P_name, core::point3f_k);
    shape.model_ = tmp.release();
    return shape;
}

shape_t shape_t::create_subd_mesh()
{
    typedef shape_model_t<subd_mesh_model_t> model_type;
    typedef core::auto_ptr_t<model_type> auto_ptr_type;

    auto_ptr_type tmp( new model_type( subd_mesh_model_t()));

    shape_t shape;
    shape.attributes().point().insert( g_P_name, core::point3f_k);
    shape.model_ = tmp.release();
    return shape;
}

shape_t shape_t::create_nurbs_curve( unsigned int degree, core::type_t point_type)
{
    typedef shape_model_t<nurbs_curve_model_t> model_type;
    typedef core::auto_ptr_t<model_type> auto_ptr_type;

    auto_ptr_type tmp( new model_type( nurbs_curve_model_t( degree)));
    shape_t shape;

    switch( point_type)
    {
        case core::point2f_k:
        case core::point3f_k:
            shape.attributes().point().insert( g_P_name, point_type);
        break;

        case core::hpoint2f_k:
        case core::hpoint3f_k:
            shape.attributes().point().insert( g_Pw_name, point_type);
        break;

        default:
            throw core::not_implemented();
    }

    shape.model_ = tmp.release();
    return shape;
}

shape_t shape_t::create_nurbs_surface( unsigned int udegree,
                                       unsigned int vdegree,
                                       unsigned int upoints,
                                       unsigned int vpoints,
                                       core::type_t point_type)
{
    typedef shape_model_t<nurbs_surface_model_t> model_type;
    typedef core::auto_ptr_t<model_type> auto_ptr_type;

    auto_ptr_type tmp( new model_type( nurbs_surface_model_t( udegree, vdegree, upoints, vpoints)));
    shape_t shape;

    unsigned int num_points = upoints * vpoints;

    switch( point_type)
    {
        case core::point3f_k:
            shape.attributes().point().insert( g_P_name, point_type);
            shape.attributes().point().array( g_P_name).reserve( num_points);
        break;

        case core::hpoint3f_k:
            shape.attributes().point().insert( g_Pw_name, point_type);
            shape.attributes().point().array( g_Pw_name).reserve( num_points);
        break;

        default:
            throw core::not_implemented();
    }

    shape.model_ = tmp.release();
    return shape;
}

shape_t shape_t::create_curves( unsigned int degree,
                                curve_basis_t basis,
                                curve_periodicity_t wrap,
                                core::type_t point_type)
{
    typedef shape_model_t<curves_model_t> model_type;
    typedef core::auto_ptr_t<model_type> auto_ptr_type;

    auto_ptr_type tmp( new model_type( curves_model_t( degree, basis, wrap)));

    shape_t shape;

    switch( point_type)
    {
        case core::point3f_k:
            shape.attributes().point().insert( g_P_name, point_type);
        break;

        case core::hpoint3f_k:
            shape.attributes().point().insert( g_Pw_name, point_type);
        break;

        default:
            throw core::not_implemented();
    }

    shape.model_ = tmp.release();
    return shape;
}

shape_t shape_t::create_points()
{
    typedef shape_model_t<points_model_t> model_type;
    typedef core::auto_ptr_t<model_type> auto_ptr_type;

    auto_ptr_type tmp( new model_type( points_model_t()));

    shape_t shape;
    shape.attributes().point().insert( g_P_name, core::point3f_k);
    shape.model_ = tmp.release();
    return shape;
}

shape_t::shape_t() : model_( 0) {}

shape_t::~shape_t() { delete model_;}

shape_t::shape_t( const shape_t& other) :   attributes_( other.attributes_),
                                            name_( other.name_)
{
    assert( other.model_);

    core::auto_ptr_t<shape_interface_t> tmp( other.model_->copy());
    model_ = tmp.release();
}

void shape_t::swap( shape_t& other)
{
    attributes_.swap( other.attributes_);
    name_.swap( other.name_);
    std::swap( model_, other.model_);
}

shape_type_t shape_t::type() const
{
    assert( model_);

    return model_->type();
}

const shape_attributes_t& shape_t::attributes() const
{
    return attributes_;
}

shape_attributes_t& shape_t::attributes()
{
    return attributes_;
}

void shape_t::accept( const_shape_visitor& v) const
{
    assert( model_);

    model_->accept( v, *this);
}

void shape_t::accept( shape_visitor& v)
{
    assert( model_);

    model_->accept( v, *this);
}

bool shape_t::check_consistency() const
{
    assert( model_);

    if( !attributes().check_consistency())
        return false;

    if( !model_->check_consistency( attributes()))
        return false;

    return true;
}

const void *shape_t::get_model() const
{
    assert( model_);

    return model_->get_model();
}

void *shape_t::get_model()
{
    assert( model_);

    return model_->get_model();
}

const core::string8_t& shape_t::name() const
{
    return name_;
}

void shape_t::set_name( const core::string8_t& n)
{
    name_ = n;
}

std::size_t shape_t::num_points() const
{
    if( !attributes().point().has_attribute( g_P_name))
        return false;

    return attributes().point().const_array( g_P_name).size();
}

void apply_visitor( const_shape_visitor& v, const shape_t& s)
{
    s.accept( v);
}

void apply_visitor( shape_visitor& v, shape_t& s)
{
    s.accept( v);
}

} // geo
} // ramen
