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

#ifndef RAMEN_MATH_TRANSFORM44_HPP
#define RAMEN_MATH_TRANSFORM44_HPP

#include<ramen/math/config.hpp>

#include<algorithm>

#include<boost/optional.hpp>

#include<ramen/math/matrix44.hpp>
#include<ramen/math/normal.hpp>

namespace ramen
{
namespace math
{

template<class T>
class transform44_t : boost::multipliable<transform44_t<T> >
{
public:

    typedef T                               value_type;
    typedef matrix44_t<T>                   matrix_type;
    typedef boost::optional<matrix44_t<T> > inverse_matrix_type;

    transform44_t()
    {
        matrix_  = matrix_type::identity_matrix();
        inverse_ = matrix_;
    }

    explicit transform44_t( const matrix_type& m)
    {
        matrix_ = m;
        inverse_ = invert( m);
    }

    transform44_t( const matrix_type& m, const matrix_type& m_inv)
    {
        matrix_  = m;
        inverse_ = m_inv;
        // TODO: check that matrix_ * inverse_ == identity;
    }

    transform44_t<T>& operator*=( const transform44_t<T>& other)
    {
        matrix_  *= other.matrix_;

        if( inverse_ && other.inverse_)
            inverse_ = inverse_.get() * other.inverse_.get();
        else
            inverse_.reset();

        return *this;
    }

    const matrix_type& matrix() const
    {
        return matrix_;
    }

    const inverse_matrix_type& inverse_matrix() const
    {
        return inverse_;
    }

    template<class S>
    vector3_t<S> transform( const vector3_t<S>& v) const
    {
        return v * matrix_;
    }

    template<class S>
    point3_t<S> transform( const point3_t<S>& p) const
    {
        return p * matrix_;
    }

    template<class S>
    hpoint3_t<S> transform( const hpoint3_t<S>& p) const
    {
        return p * matrix_;
    }

    template<class S>
    normal_t<S> transform( const normal_t<S>& n) const
    {
        if( inverse_)
        {
            const matrix_type& inv = inverse_.get();
            return normal_t<S>( n.x * inv( 0, 0) + n.y * inv( 0, 1) + n.z * inv( 0, 2),
                                n.x * inv( 1, 0) + n.y * inv( 1, 1) + n.z * inv( 1, 2),
                                n.x * inv( 2, 0) + n.y * inv( 2, 1) + n.z * inv( 2, 2));
        }

        return normal_t<S>( 0, 0, 0);
    }

    template<class S>
    vector3_t<S> inverse_transform( const vector3_t<S>& v) const
    {
        if( inverse_)
            return v * inverse_.get();

        return vector3_t<S>( 0, 0, 0);
    }

    template<class S>
    point3_t<S> inverse_transform( const point3_t<S>& p) const
    {
        if( inverse_)
            return p * inverse_.get();

        return point3_t<S>( 0, 0, 0);
    }

    template<class S>
    hpoint3_t<S> inverse_transform( const hpoint3_t<S>& p) const
    {
        if( inverse_)
            return p * inverse_.get();

        return hpoint3_t<S>( 0, 0, 0);
    }

    template<class S>
    normal_t<S> inverse_transform( const normal_t<S>& n) const
    {
        return normal_t<S>( n.x * matrix_( 0, 0) + n.y * matrix_( 0, 1) + n.z * matrix_( 0, 2),
                            n.x * matrix_( 1, 0) + n.y * matrix_( 1, 1) + n.z * matrix_( 1, 2),
                            n.x * matrix_( 2, 0) + n.y * matrix_( 2, 1) + n.z * matrix_( 2, 2));
    }

    static transform44_t<T> zero_transform()
    {
        transform44_t<T> result;
        result.matrix_ = matrix_type::zero_matrix();
        result.inverse_.reset();
    }

private:

    matrix_type matrix_;
    inverse_matrix_type inverse_;
};

template<class T>
transform44_t<T> invert( const transform44_t<T>& m)
{
    if( m.inverse_matrix())
        return transform44_t<T>( m.inverse_matrix().get(), m.matrix());
    else
        return transform44_t<T>::zero_transform();
}

typedef transform44_t<float>   transform44f_t;
typedef transform44_t<double>  transform44d_t;

} // math
} // ramen

#endif
