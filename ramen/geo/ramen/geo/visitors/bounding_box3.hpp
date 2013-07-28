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

#ifndef RAMEN_GEO_ALGORITHM_BOUNDING_BOX_HPP
#define RAMEN_GEO_ALGORITHM_BOUNDING_BOX_HPP

#include<ramen/geo/shape_models/shape_visitor.hpp>

#include<ramen/core/exceptions.hpp>
#include<ramen/core/name.hpp>

#include<ramen/math/box3.hpp>

#include<ramen/arrays/apply_visitor.hpp>

#include<ramen/geo/shape.hpp>
#include<ramen/geo/global_names.hpp>

namespace ramen
{
namespace geo
{
namespace detail
{

template<class T>
class bbox3_visitor
{
public:

    explicit bbox3_visitor( math::box3_t<T>& box) : box_( box) {}

    template<class X>
    void operator()( const arrays::const_array_ref_t<math::point3_t<X> >& array)
    {
        for( int i = 0, e = array.size(); i < e; ++i)
        {
            const math::point3_t<X>& p( array[i]);
            box_.extend_by( math::point3_t<T>( p.x, p.y, p.z));
        }
    }

    // special case for homogeneus points.
    template<class X>
    void operator()( const arrays::const_array_ref_t<math::hpoint3_t<X> >& array)
    {
        for( int i = 0, e = array.size(); i < e; ++i)
        {
            const math::hpoint3_t<X>& p( array[i]);

            if( p.w <= X( 0))
                throw core::runtime_error( "Zero or negative weight found in bounding_box3_visitor");

            box_.extend_by( math::point3_t<T>( p.x / p.w,
                                               p.y / p.w,
                                               p.z / p.w));
        }
    }

    // catch all
    template<class Any>
    void operator()( const arrays::const_array_ref_t<Any>& array)
    {
        throw core::runtime_error( core::make_string( "Bad array type: ",
                                                      core::type_to_string( core::type_traits<Any>::type()),
                                                      " in bounding_box3 visitor"));
    }

private:

   math::box3_t<T>& box_;
};

} // detail

template<class T = double>
class bounding_box3_visitor : public const_shape_visitor
{
public:

    bounding_box3_visitor( const core::name_t& point_attr_name = core::name_t(),
                           const math::box3_t<T>& box = math::box3_t<T>())
    {
        init( point_attr_name, box);
    }

    explicit bounding_box3_visitor( const math::box3_t<T>& box) : const_shape_visitor()
    {
        init( core::name_t(), box);
    }

    virtual void visit( const visitable_t& model, const shape_t& shape)
    {
        if( !point_attr_name_.empty())
        {
            if( shape.attributes().point().has_attribute( point_attr_name_))
                calc_bbox( shape.attributes().point().const_array( point_attr_name_));
            else
                throw core::runtime_error( core::make_string( "No ", point_attr_name_.c_str(),
                                                  " attribute found in bounding_box3_visitor"));
        }
        else
        {
            if( shape.attributes().point().has_attribute( g_P_name))
                calc_bbox( shape.attributes().point().const_array( g_P_name));
            else
            {
                if ( shape.attributes().point().has_attribute( g_Pw_name))
                    calc_bbox( shape.attributes().point().const_array( g_Pw_name));
                else
                    throw core::runtime_error( "No P or Pw attribute found in bounding_box3_visitor");
            }
        }
    }

    math::box3_t<T> bbox;

private:

    void init( const core::name_t& point_attr_name, const math::box3_t<T>& box)
    {
        point_attr_name_ = point_attr_name;
        bbox = box;
    }

    void calc_bbox( const arrays::array_t& array)
    {
        detail::bbox3_visitor<T> v( bbox);
        arrays::apply_visitor( v, array);
    }

    core::name_t point_attr_name_;
};

} // geo
} // ramen

#endif
