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

#ifndef RAMEN_MATH_RAY_HPP
#define RAMEN_MATH_RAY_HPP

#include<ramen/math/config.hpp>

#include<algorithm>

#include<boost/mpl/same_as.hpp>
#include<boost/mpl/if.hpp>

#include<ramen/core/Concepts/RegularConcept.hpp>

#include<ramen/math/box3.hpp>
#include<ramen/math/matrix44.hpp>

namespace ramen
{
namespace math
{
namespace detail
{

template<class T>
class ray_base_t
{
    BOOST_CONCEPT_ASSERT(( core::RegularConcept<T>));

public:

    const T& data() const   { return data_;}
    T& data()               { return data_;}

protected:

    T data_;
};

template<>
struct ray_base_t<void> {};

} // detail

/*!
\brief ray.
*/
template<class T, class Data = void>
class ray_t : public detail::ray_base_t<Data>
{
public:

    BOOST_STATIC_CONSTANT( T, epsilon_k = 1.0e-3);
    BOOST_STATIC_CONSTANT( T, maxt_k = 1.0e30);

    ray_t()
    {
        set_origin( point3_t<T>::origin());
        set_direction( vector3_t<T>( 0, 0, 1));
        set_mint( epsilon_k);
        set_maxt( maxt_k);
        set_time( 0);
    }

    ray_t( const point3_t<T>& origin,
           const vector3_t<T>& direction,
           T maxt = T( maxt_k),
           T time = T(0))
    {
        set_origin( origin);
        set_direction( direction);
        set_mint( epsilon_k);
        set_maxt( maxt);
        set_time( time);
    }

    const point3_t<T>& origin() const       { return origin_;}
    void set_origin( const point3_t<T>& p)  { origin_ = p;}

    const vector3_t<T>& direction() const   { return direction_;}

    void set_direction( const vector3_t<T>& d)
    {
        direction_ = normalize( d);
        rcp_direction_ = vector3_t<T>( T(1) / d.x, T(1) / d.y, T(1) / d.z);
    }

    const vector3_t<T>& reciprocal_direction() const
    {
        return rcp_direction_;
    }

    T mint() const { return mint_;}

    void set_mint( T x)
    {
        assert( x > T(0));

        mint_ = x;
    }

    T maxt() const { return maxt_;}

    void set_maxt( T x)
    {
        assert( x > T(0));

        maxt_ = x;
    }

    T time() const      { return time_;}
    void set_time( T x) { time_ = x;}

    point3_t<T> operator()( float t) const
    {
        return origin_ + t * direction_;
    }

    template<class S>
    void transform( const matrix44_t<S>& m)
    {
        set_origin( m * origin_);
        set_direction( m * direction_);
    }

    bool intersect( const box3_t<T>& box, T& tmin, T& tmax) const
    {
        if( box.empty())
            return false;

        T t0 = mint();
        T t1 = maxt();

        for( int i = 0; i < 3; ++i)
        {
            T tnear = ( box.min( i) - origin_( i)) * rcp_direction_( i);
            T tfar  = ( box.max( i) - origin_( i)) * rcp_direction_( i);

            if( tnear > tfar)
                std::swap( tnear, tfar);

            t0 = tnear > t0 ? tnear : t0;
            t1 = tfar  < t1 ? tfar  : t1;

            if( t0 > t1)
                return false;
        }

        tmin = t0;
        tmax = t1;
        return true;
    }

private:

    point3_t<T> origin_;
    vector3_t<T> direction_;
    T mint_;
    T maxt_;
    T time_;

    // precalculated, for ray / box intersections.
    vector3_t<T> rcp_direction_;
};

template<class T, class S>
ray_t<T> transform( const ray_t<T>& r, const matrix44_t<S>& m)
{
    ray_t<T> result( r);
    result.transform( m);
    return result;
}

typedef ray_t<float>     rayf_t;
typedef ray_t<double>    rayd_t;

} // math
} // ramen

#endif
