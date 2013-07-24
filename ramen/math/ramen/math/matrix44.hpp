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

#ifndef RAMEN_MATH_MATRIX44_HPP
#define RAMEN_MATH_MATRIX44_HPP

#include<ramen/math/config.hpp>

#include<algorithm>

#include<boost/operators.hpp>
#include<boost/optional.hpp>

#include<ramen/math/cmath.hpp>
#include<ramen/math/constants.hpp>
#include<ramen/math/hpoint3.hpp>

namespace ramen
{
namespace math
{

/*!
\brief A 4x4 matrix.
*/
template<class T>
class matrix44_t :  boost::multipliable<matrix44_t<T>,
                    boost::equality_comparable<matrix44_t<T> > >
{
public:

    typedef T           value_type;
    typedef const T*    const_iterator;
    typedef T*          iterator;

    static unsigned int	rows()  { return 4;}
    static unsigned int	cols()  { return 4;}
    static unsigned int	size()  { return 16;}

    matrix44_t() {}

    matrix44_t( T a00, T a01, T a02, T a03,
                T a10, T a11, T a12, T a13,
                T a20, T a21, T a22, T a23,
                T a30, T a31, T a32, T a33)
    {
        data_[0][0] = a00; data_[0][1] = a01; data_[0][2] = a02; data_[0][3] = a03;
        data_[1][0] = a10; data_[1][1] = a11; data_[1][2] = a12; data_[1][3] = a13;
        data_[2][0] = a20; data_[2][1] = a21; data_[2][2] = a22; data_[2][3] = a23;
        data_[3][0] = a30; data_[3][1] = a31; data_[3][2] = a32; data_[3][3] = a33;
    }

    template<class Iter>
    explicit matrix44_t( Iter it)
    {
        std::copy( it, it + size(), begin());
    }

    T operator()( unsigned int j, unsigned int i) const
    {
        assert( j < rows());
        assert( i < cols());

        return data_[j][i];
    }

    T& operator()( unsigned int j, unsigned int i)
    {
        assert( j < rows());
        assert( i < cols());

        return data_[j][i];
    }

    // iterators
    const_iterator begin() const    { return &data_[0][0];}
    const_iterator end() const      { return begin() + size();}

    iterator begin()    { return &data_[0][0];}
    iterator end()      { return begin() + size();}

    // operators
    matrix44_t<T>& operator*=( const matrix44_t<T>& other)
    {
        matrix44_t<T> tmp( zero_matrix());

        for( unsigned int i = 0; i < rows(); i++)
            for( unsigned int j = 0; j < cols(); j++)
                for( unsigned int k = 0; k < rows(); k++)
                    tmp( i, j) += data_[i][k] * other( k, j);

        *this = tmp;
        return *this;
    }

    void transpose()
    {
        matrix44_t<T> tmp;
        for( int i = 0; i < rows(); i++)
            for( int j = 0; j < cols(); j++)
                tmp( i, j) = (*this)( j, i);

        *this = tmp;
    }

    bool operator==( const matrix44_t<T>& other) const
    {
        return std::equal( begin(), end(), other.begin(), other.end());
    }

    // static factory methods
    static matrix44_t<T> identity_matrix()
    {
        return matrix44_t<T>(   1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0,
                                0, 0, 0, 1);
    }

    static matrix44_t<T> zero_matrix()
    {
        return matrix44_t<T>(   0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0);
    }

    static matrix44_t<T> translation_matrix( const vector3_t<T>& t)
    {
        return matrix44_t<T>(   1  , 0  , 0  , 0,
                                0  , 1  , 0  , 0,
                                0  , 0  , 1  , 0,
                                t.x, t.y, t.z, 1);
    }

    static matrix44_t<T> scale_matrix( T s)
    {
        return matrix44_t<T>(   s, 0, 0, 0,
                                0, s, 0, 0,
                                0, 0, s, 0,
                                0, 0, 0, 1);
    }

    static matrix44_t<T> scale_matrix( const vector3_t<T>& s)
    {
        return matrix44_t<T>(   s.x, 0  , 0  , 0,
                                0  , s.y, 0  , 0,
                                0  , 0  , s.z, 0,
                                0  , 0  , 0  , 1);
    }

    static matrix44_t<T> axis_angle_rotation_matrix( vector3_t<T> axis, T angle_in_deg)
    {
        vector3_t<T> unit( axis.normalized());
        T angle_in_rad = angle_in_deg * constants<T>::deg2rad();
        T sine   = cmath<T>::sin( angle_in_rad);
        T cosine = cmath<T>::cos( angle_in_rad);

        matrix44_t<T> m;
        m( 0, 0) = unit( 0) * unit( 0) * (1 - cosine) + cosine;
        m( 0, 1) = unit( 0) * unit( 1) * (1 - cosine) + unit( 2) * sine;
        m( 0, 2) = unit( 0) * unit( 2) * (1 - cosine) - unit( 1) * sine;
        m( 0, 3) = 0;

        m( 1, 0) = unit( 0) * unit( 1) * (1 - cosine) - unit( 2) * sine;
        m( 1, 1) = unit( 1) * unit( 1) * (1 - cosine) + cosine;
        m( 1, 2) = unit( 1) * unit( 2) * (1 - cosine) + unit( 0) * sine;
        m( 1, 3) = 0;

        m( 2, 0) = unit( 0) * unit( 2) * (1 - cosine) + unit( 1) * sine;
        m( 2, 1) = unit( 1) * unit( 2) * (1 - cosine) - unit( 0) * sine;
        m( 2, 2) = unit( 2) * unit( 2) * (1 - cosine) + cosine;
        m( 2, 3) = 0;

        m( 3, 0) = 0;
        m( 3, 1) = 0;
        m( 3, 2) = 0;
        m( 3, 3) = 1;
        return m;
    }

    static matrix44_t<T> rotate_x_matrix( T angle_in_deg)
    {
        return axis_angle_rotation_matrix( vector3_t<T>( 1, 0, 0), angle_in_deg);
    }

    static matrix44_t<T> rotate_y_matrix( T angle_in_deg)
    {
        return axis_angle_rotation_matrix( vector3_t<T>( 0, 1, 0), angle_in_deg);
    }

    static matrix44_t<T> rotate_z_matrix( T angle_in_deg)
    {
        return axis_angle_rotation_matrix( vector3_t<T>( 0, 0, 1), angle_in_deg);
    }

    static matrix44_t<T> align_z_axis_with_target_dir_matrix( vector3_t<T> target_dir,
                                                              vector3_t<T> up_dir)
    {
        matrix44_t<T> result;

        // Ensure that the target direction is non-zero.
        if( target_dir.length () == 0)
            target_dir = vector3_t<T>( 0, 0, 1);

        // Ensure that the up direction is non-zero.
        if( up_dir.length() == 0)
            up_dir = vector3_t<T>( 0, 1, 0);

        //
        // Check for degeneracies.  If the upDir and targetDir are parallel
        // or opposite, then compute a new, arbitrary up direction that is
        // not parallel or opposite to the targetDir.
        //

        if( cross( up_dir, target_dir).length() == 0)
        {
            up_dir = cross( target_dir, vector3_t<T> (1, 0, 0));

            if( up_dir.length() == 0)
                up_dir = cross( target_dir, vector3_t<T> (0, 0, 1));
        }

        // Compute the x-, y-, and z-axis vectors of the new coordinate system.
        vector3_t<T> target_perp_dir = cross( up_dir, target_dir);
        vector3_t<T> target_up_dir   = cross( target_dir, target_perp_dir);

        // Rotate the x-axis into targetPerpDir (row 0),
        // rotate the y-axis into targetUpDir   (row 1),
        // rotate the z-axis into targetDir     (row 2).
        vector3_t<T> row[3];
        row[0] = target_perp_dir.normalized();
        row[1] = target_up_dir.normalized();
        row[2] = target_dir.normalized();

        result( 0, 0) = row[0]( 0);
        result( 0, 1) = row[0]( 1);
        result( 0, 2) = row[0]( 2);
        result( 0, 3) = (T) 0;

        result( 1, 0) = row[1]( 0);
        result( 1, 1) = row[1]( 1);
        result( 1, 2) = row[1]( 2);
        result( 1, 3) = (T) 0;

        result( 2, 0) = row[2]( 0);
        result( 2, 1) = row[2]( 1);
        result( 2, 2) = row[2]( 2);
        result( 2, 3) = (T) 0;

        result( 3, 0) = (T) 0;
        result( 3, 1) = (T) 0;
        result( 3, 2) = (T) 0;
        result( 3, 3) = (T) 1;
        return result;
    }

    static matrix44_t<T> rotation_with_up_dir_matrix( const vector3_t<T>& from_dir,
                                                      const vector3_t<T>& to_dir,
                                                      const vector3_t<T>& up_dir)
    {
        //
        // The goal is to obtain a rotation matrix that takes
        // "fromDir" to "toDir".  We do this in two steps and
        // compose the resulting rotation matrices;
        //    (a) rotate "fromDir" into the z-axis
        //    (b) rotate the z-axis into "toDir"
        //

        // The from direction must be non-zero; but we allow zero to and up dirs.
        if( from_dir.length() == 0)
            return matrix44_t<T>();
        else
        {
            matrix44_t<T> z_axis_to_from_dir;
            z_axis_to_from_dir = align_z_axis_with_target_dir_matrix( from_dir, vector3_t<T>( 0, 1, 0));

            matrix44_t<T> from_dir_to_z_axis = z_axis_to_from_dir.transposed();
            matrix44_t<T> z_axis_to_to_dir = align_z_axis_with_target_dir_matrix( to_dir, up_dir);
            return from_dir_to_z_axis * z_axis_to_to_dir;
        }
    }

    static matrix44_t<T> rotation_matrix( const vector3_t<T>& r)
    {
        T cos_rz = cmath<T>::cos( r(2) * constants<T>::deg2rad());
        T cos_ry = cmath<T>::cos( r(1) * constants<T>::deg2rad());
        T cos_rx = cmath<T>::cos( r(0) * constants<T>::deg2rad());
        T sin_rz = cmath<T>::sin( r(2) * constants<T>::deg2rad());
        T sin_ry = cmath<T>::sin( r(1) * constants<T>::deg2rad());
        T sin_rx = cmath<T>::sin( r(0) * constants<T>::deg2rad());

        matrix44_t<T> result;
        result( 0, 0) =  cos_rz * cos_ry;
        result( 0, 1) =  sin_rz * cos_ry;
        result( 0, 2) = -sin_ry;
        result( 0, 3) =  0;

        result( 1, 0) = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
        result( 1, 1) =  cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
        result( 1, 2) =  cos_ry * sin_rx;
        result( 1, 3) =  0;

        result( 2, 0) =  sin_rz * sin_rx + cos_rz * sin_ry * cos_rx;
        result( 2, 1) = -cos_rz * sin_rx + sin_rz * sin_ry * cos_rx;
        result( 2, 2) =  cos_ry * cos_rx;
        result( 2, 3) =  0;

        result( 3, 0) =  0;
        result( 3, 1) =  0;
        result( 3, 2) =  0;
        result( 3, 3) =  1;
        return result;
    }

private:

    T data_[4][4];
};

template<class T, class S>
vector3_t<S> operator*( const vector3_t<S>& v, const matrix44_t<T>& m)
{
    return vector3_t<S>( v.x * m( 0, 0) + v.y * m( 1, 0) + v.z * m( 2, 0),
                         v.x * m( 0, 1) + v.y * m( 1, 1) + v.z * m( 2, 1),
                         v.x * m( 0, 2) + v.y * m( 1, 2) + v.z * m( 2, 2));
}

template<class T, class S>
point3_t<S> operator*( const point3_t<S>& p, const matrix44_t<T>& m)
{
    S w = p.x * m( 0, 3) + p.y * m( 1, 3) + p.z * m( 2, 3) + m( 3, 3);
    return point3_t<S>( (p.x * m( 0, 0) + p.y * m( 1, 0) + p.z * m( 2, 0) + m( 3, 0)) / w,
                        (p.x * m( 0, 1) + p.y * m( 1, 1) + p.z * m( 2, 1) + m( 3, 1)) / w,
                        (p.x * m( 0, 2) + p.y * m( 1, 2) + p.z * m( 2, 2) + m( 3, 2)) / w);
}

template<class T, class S>
hpoint3_t<S> operator*( const hpoint3_t<S>& p, const matrix44_t<T>& m)
{
    return hpoint3_t<S>( p.x * m( 0, 0) + p.y * m( 1, 0) + p.z * m( 2, 0) + p.w * m( 3, 0),
                         p.x * m( 0, 1) + p.y * m( 1, 1) + p.z * m( 2, 1) + p.w * m( 3, 1),
                         p.x * m( 0, 2) + p.y * m( 1, 2) + p.z * m( 2, 2) + p.w * m( 3, 2),
                         p.x * m( 0, 3) + p.y * m( 1, 3) + p.z * m( 2, 3) + p.w * m( 3, 3));
}

template<class T>
matrix44_t<T> transpose( const matrix44_t<T>& m)
{
    matrix44_t<T> x( m);
    x.transpose();
    return x;
}

template<class T>
boost::optional<matrix44_t<T> > invert_gauss_jordan( const matrix44_t<T>& m)
{
    int i, j, k;
    matrix44_t<T> s;
    matrix44_t<T> t( m);

    // Forward elimination
    for( i = 0; i < 3 ; ++i)
    {
        int pivot = i;
        T pivotsize = t( i, i);

        if( pivotsize < 0)
            pivotsize = -pivotsize;

        for( j = i + 1; j < 4; ++j)
        {
            T tmp = t( j, i);

            if( tmp < 0)
                tmp = -tmp;

            if( tmp > pivotsize)
            {
                pivot = j;
                pivotsize = tmp;
            }
        }

        if( pivotsize == 0)
            return boost::optional<matrix44_t<T> >();

        if( pivot != i)
        {
            for( j = 0; j < 4; ++j)
            {
                T tmp;

                tmp = t( i, j);
                t( i, j) = t( pivot, j);
                t( pivot, j) = tmp;

                tmp = s( i, j);
                s( i, j) = s( pivot, j);
                s( pivot, j) = tmp;
            }
        }

        for( j = i + 1; j < 4; ++j)
        {
            T f = t( j, i) / t( i, i);

            for( k = 0; k < 4; ++k)
            {
                t( j, k) -= f * t( i, k);
                s( j, k) -= f * s( i, k);
            }
        }
    }

    // Backward substitution
    for( i = 3; i >= 0; --i)
    {
        T f;

        if( (f = t( i, i)) == 0)
            return boost::optional<matrix44_t<T> >();

        for(j = 0; j < 4; j++)
        {
            t( i, j) /= f;
            s( i, j) /= f;
        }

        for(j = 0; j < i; j++)
        {
            f = t( j, i);

            for(k = 0; k < 4; k++)
            {
                t( j, k) -= f * t( i, k);
                s( j, k) -= f * s( i, k);
            }
        }
    }

    return s;
}

template<class T>
boost::optional<matrix44_t<T> > invert( const matrix44_t<T>& m)
{
    if( m( 0,3) != 0 || m( 1,3) != 0 || m( 2,3) != 0 || m( 3,3) != 1)
        return invert_gauss_jordan( m);

    matrix44_t<T> s( m( 1,1) * m( 2,2) - m( 2,1) * m( 1,2),
                     m( 2,1) * m( 0,2) - m( 0,1) * m( 2,2),
                     m( 0,1) * m( 1,2) - m( 1,1) * m( 0,2),
                     0,

                     m( 2,0) * m( 1,2) - m( 1,0) * m( 2,2),
                     m( 0,0) * m( 2,2) - m( 2,0) * m( 0,2),
                     m( 1,0) * m( 0,2) - m( 0,0) * m( 1,2),
                     0,

                     m( 1,0) * m( 2,1) - m( 2,0) * m( 1,1),
                     m( 2,0) * m( 0,1) - m( 0,0) * m( 2,1),
                     m( 0,0) * m( 1,1) - m( 1,0) * m( 0,1),
                     0,

                     0,
                     0,
                     0,
                     1);

    T r = m( 0,0) * s( 0, 0) + m( 0,1) * s( 1, 0) + m( 0,2) * s( 2, 0);

    if( cmath<T>::fabs( r) >= T(1))
    {
        for( int i = 0; i < 3; ++i)
        {
            for( int j = 0; j < 3; ++j)
                s( i, j) /= r;
        }
    }
    else
    {
        T mr = cmath<T>::fabs( r) / std::numeric_limits<T>::min();

        for( int i = 0; i < 3; ++i)
        {
            for( int j = 0; j < 3; ++j)
            {
                if( mr > cmath<T>::fabs( s( i, j)))
                    s( i, j) /= r;
                else
                    return boost::optional<matrix44_t<T> >();
            }
        }
    }

    s( 3, 0) = -m( 3,0) * s( 0, 0) - m( 3,1) * s( 1, 0) - m( 3,2) * s( 2, 0);
    s( 3, 1) = -m( 3,0) * s( 0, 1) - m( 3,1) * s( 1, 1) - m( 3,2) * s( 2, 1);
    s( 3, 2) = -m( 3,0) * s( 0, 2) - m( 3,1) * s( 1, 2) - m( 3,2) * s( 2, 2);
    return s;
}

typedef matrix44_t<float>   matrix44f_t;
typedef matrix44_t<double>  matrix44d_t;

#ifdef RAMEN_WITH_HALF
    typedef matrix44_t<half> matrix44h_t;
#endif

} // math
} // ramen

#endif
