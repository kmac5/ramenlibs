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

#ifndef RAMEN_GEO_SHAPE_HPP
#define RAMEN_GEO_SHAPE_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_fwd.hpp>

#include<ramen/math/box3.hpp>

#include<ramen/geo/shape_attributes.hpp>

#include<ramen/geo/shape_models/shape_types.hpp>
#include<ramen/geo/shape_models/shape_visitor.hpp>
#include<ramen/geo/shape_models/poly_mesh_model_fwd.hpp>
#include<ramen/geo/shape_models/subd_mesh_model_fwd.hpp>
#include<ramen/geo/shape_models/nurbs_curve_model_fwd.hpp>
#include<ramen/geo/shape_models/nurbs_surface_model_fwd.hpp>
#include<ramen/geo/shape_models/curves_model_fwd.hpp>
#include<ramen/geo/shape_models/points_model_fwd.hpp>

namespace ramen
{
namespace geo
{

/*!
\brief Shape class.
*/
class RAMEN_GEO_API shape_t
{

    BOOST_COPYABLE_AND_MOVABLE( shape_t)

public:

    static shape_t create_poly_mesh();
    static shape_t create_subd_mesh();
    static shape_t create_nurbs_curve( unsigned int degree, core::type_t point_type = core::hpoint3f_k);
    static shape_t create_nurbs_surface( unsigned int udegree,
                                         unsigned int vdegree,
                                         unsigned int upoints,
                                         unsigned int vpoints,
                                         core::type_t point_type = core::hpoint3f_k);

    static shape_t create_curves( unsigned int degree,
                                  curve_basis_t basis,
                                  curve_periodicity_t wrap,
                                  core::type_t point_type = core::point3f_k);

    static shape_t create_points();

    ~shape_t();

    // Copy constructor
    shape_t( const shape_t& other);

    // Copy assignment
    shape_t& operator=( BOOST_COPY_ASSIGN_REF( shape_t) other)
    {
        shape_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    shape_t( BOOST_RV_REF( shape_t) other)
    {
        assert( other.model_);

        model_ = 0;
        swap( other);
    }

    // Move assignment
    shape_t& operator=( BOOST_RV_REF( shape_t) other)
    {
        assert( other.model_);

        swap( other);
        return *this;
    }

    void swap( shape_t& other);

    shape_type_t type() const;

    const shape_attributes_t& attributes() const;
    shape_attributes_t& attributes();

    // check if the shape is consistent.
    bool check_consistency() const;

    const core::string8_t& name() const;
    void set_name( const core::string8_t& n);

    std::size_t num_points() const;

    // has to be public, to allow subclassing of shape_interface_t,
    // inside shape.cpp. Treat it as if private.
    struct shape_interface_t;

private:

    const void *get_model() const;
    void *get_model();

    template<class T>
    friend const T& shape_cast( const shape_t& s);

    template<class T>
    friend T& shape_cast( shape_t& s);

    template<class T>
    friend const T *shape_cast( const shape_t *s);

    template<class T>
    friend T *shape_cast( shape_t *s);

    friend void apply_visitor( const_shape_visitor& v, const shape_t& s);
    friend void apply_visitor( shape_visitor& v, shape_t& s);

    shape_t();

    // visitor
    void accept( const_shape_visitor& v) const;
    void accept( shape_visitor& v);

    core::string8_t name_;
    shape_attributes_t attributes_;

    shape_interface_t *model_;
};

inline void swap( shape_t& x, shape_t& y)
{
    x.swap( y);
}

template<class T>
const T& shape_cast( const shape_t& s)
{
    if( shape_model_traits<T>::type() != s.type())
        throw bad_shape_cast( s.type(), shape_model_traits<T>::type());

    return *(reinterpret_cast<const T*>( s.get_model()));
}

template<class T>
T& shape_cast( shape_t& s)
{
    if( shape_model_traits<T>::type() != s.type())
        throw bad_shape_cast( s.type(), shape_model_traits<T>::type());

    return *( reinterpret_cast<T*>( s.get_model()));
}

template<class T>
const T *shape_cast( const shape_t *s)
{
    if( shape_model_traits<T>::type() != s->type())
        return 0;

    return reinterpret_cast<const T*>( s->get_model());
}

template<class T>
T *shape_cast( shape_t *s)
{
    if( shape_model_traits<T>::type() != s->type())
        return 0;

    return reinterpret_cast<T*>( s->get_model());
}

RAMEN_GEO_API void apply_visitor( const_shape_visitor& v, const shape_t& s);
RAMEN_GEO_API void apply_visitor( shape_visitor& v, shape_t& s);

} // geo
} // ramen

#endif
