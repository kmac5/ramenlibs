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

#ifndef RAMEN_GEO_ALGORITHM_TRANSFORM_HPP
#define RAMEN_GEO_ALGORITHM_TRANSFORM_HPP

#include<ramen/geo/shape_models/shape_visitor.hpp>

#include<ramen/math/transform44.hpp>

#include<ramen/arrays/apply_visitor.hpp>

#include<ramen/geo/shape.hpp>

namespace ramen
{
namespace geo
{
namespace detail
{

template<class T>
class transform_array
{
public:

    explicit transform_array( const math::transform44_t<T>& m) : m_( m) {}

    template<class X>
    void operator()( arrays::array_ref_t<math::point3_t<X> >& array) const
    {
        for( int i = 0, e = array.size(); i < e; ++i)
            array[i] = m_.transform( array[i]);
    }

    template<class X>
    void operator()( arrays::array_ref_t<math::hpoint3_t<X> >& array) const
    {
        for( int i = 0, e = array.size(); i < e; ++i)
            array[i] = m_.transform( array[i]);
    }

    template<class X>
    void operator()( arrays::array_ref_t<math::vector3_t<X> >& array) const
    {
        for( int i = 0, e = array.size(); i < e; ++i)
            array[i] = m_.transform( array[i]);
    }

    template<class X>
    void operator()( arrays::array_ref_t<math::normal_t<X> >& array) const
    {
        for( int i = 0, e = array.size(); i < e; ++i)
            array[i] = m_.transform( array[i]);
    }

    template<class Any>
    void operator()( arrays::array_ref_t<Any>& array) const
    {
        // skip things that don't need transformation, like colors, scalars, ...
    }

private:

    const math::transform44_t<T>& m_;
};

template<class T>
void transform_attribute_table( attribute_table_t& table,
                                const math::transform44_t<T>& m)
{
    transform_array<T> v( m);
    for( attribute_table_t::iterator it( table.begin()), e( table.end()); it != e; ++it)
        arrays::apply_visitor( v, it.second());
}

template<class T>
void transform_dictionary( core::dictionary_t& dict,
                           const math::transform44_t<T>& m)
{
    for( core::dictionary_t::iterator it( dict.begin()), e( dict.end()); it != e; ++it)
    {
        switch( it->second.type())
        {
            case core::point3f_k:
            {
                math::point3f_t *p = core::get<math::point3f_t>( &it->second);
                it->second = core::variant_t( m.transform( *p));
            }
            break;

            case core::hpoint3f_k:
            {
                math::hpoint3f_t *p = core::get<math::hpoint3f_t>( &it->second);
                it->second = core::variant_t( m.transform( *p));
            }
            break;

            default:
                // skip
            break;
        }
    }
}

} // detail

template<class T>
class transform_visitor : public shape_visitor
{
public:

    explicit transform_visitor( const math::matrix44_t<T>& m) : xform_( m) {}
    explicit transform_visitor( const math::transform44_t<T>& xform) : xform_( xform) {}

    virtual void visit( visitable_t& model, shape_t& shape)
    {
        detail::transform_dictionary( shape.attributes().constant() , xform_);
        detail::transform_attribute_table( shape.attributes().primitive(), xform_);
        detail::transform_attribute_table( shape.attributes().vertex()   , xform_);
        detail::transform_attribute_table( shape.attributes().point()    , xform_);
    }

private:

    const math::transform44_t<T>& xform_;
};

} // geo
} // ramen

#endif
