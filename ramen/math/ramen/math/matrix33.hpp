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

#ifndef RAMEN_MATH_MATRIX33_HPP
#define RAMEN_MATH_MATRIX33_HPP

#include<ramen/math/config.hpp>

#include<algorithm>

#include<boost/operators.hpp>
#include<boost/optional.hpp>

#include<ramen/math/hpoint2.hpp>

namespace ramen
{
namespace math
{

/*!
\brief A 3x3 matrix.
*/
template<class T>
class matrix33_t :  boost::multipliable<matrix33_t<T>,
                    boost::equality_comparable<matrix33_t<T> > >
{
public:

    typedef T           value_type;
    typedef const T*    const_iterator;
    typedef T*          iterator;

    typedef boost::mpl::int_<3> row_size_type;
    typedef boost::mpl::int_<3> col_size_type;
    typedef boost::mpl::int_<9> size_type;

    matrix33_t() {}

    matrix33_t( T a00, T a01, T a02,
                T a10, T a11, T a12,
                T a20, T a21, T a22)
    {
        data_[0][0] = a00; data_[0][1] = a01; data_[0][2] = a02;
        data_[1][0] = a10; data_[1][1] = a11; data_[1][2] = a12;
        data_[2][0] = a20; data_[2][1] = a21; data_[2][2] = a22;
    }

    matrix33_t( const T *data)
    {
        assert( data);

        std::copy( data, data + size_type::value, &data_[0][0]);
    }

    // iterators
    const_iterator begin() const    { return &data_[0][0];}
    const_iterator end() const      { return begin() + size_type::value;}

    iterator begin()    { return &data_[0][0];}
    iterator end()      { return begin() + size_type::value;}

    T operator()( unsigned int j, unsigned int i) const
    {
        assert( j < row_size_type::value);
        assert( i < col_size_type::value);

        return data_[j][i];
    }

    T& operator()( unsigned int j, unsigned int i)
    {
        assert( j < row_size_type::value);
        assert( i < col_size_type::value);

        return data_[j][i];
    }

    // operators
    matrix33_t<T>& operator*=( const matrix33_t<T>& other)
    {
        matrix33_t<T> tmp( zero_matrix());

        for( int i = 0; i < 3; i++)
            for( int j = 0; j < 3; j++)
                for( int k = 0; k < 3; k++)
                    tmp( i, j) += data_[i][k] * other( k, j);

        *this = tmp;
        return *this;
    }

    // internal access, const only
    const T *data() const
    {
        return &data_[0][0];
    }

    // for regular concept
    bool operator==( const matrix33_t<T>& other) const
    {
        return std::equal( begin(), end(), other.begin(), other.end());
    }

    static matrix33_t<T> identity_matrix()
    {
        return matrix33_t<T>( 1, 0, 0,
                              0, 1, 0,
                              0, 0, 1);
    }

    static matrix33_t<T> zero_matrix()
    {
        return matrix33_t<T>( 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0);
    }

    static matrix33_t<T> translation_matrix( const vector2_t<T>& t)
    {
        return matrix33_t<T>( 1  , 0  , 0,
                              0  , 1  , 0,
                              t.x, t.y, 1);
    }

    static matrix33_t<T> scale_matrix( T s)
    {
        return matrix33_t<T>( s, 0, 0,
                              0, s, 0,
                              0, 0, 1);
    }

    static matrix33_t<T> scale_matrix( const vector2_t<T>& s)
    {
        return matrix33_t<T>( s.x, 0  , 0,
                              0  , s.y, 0,
                              0  , 0  , 1);
    }

private:

    T data_[3][3];
};

template<class T, class S>
vector2_t<S> operator*( const vector2_t<S>& v, const matrix33_t<T>& m)
{
    return vector2_t<S>( v.x * m( 0, 0) + v.y * m( 1, 0),
                         v.x * m( 0, 1) + v.y * m( 1, 1));
}

template<class T, class S>
point2_t<S> operator*( const point2_t<S>& p, const matrix33_t<T>& m)
{
    S w = p.x * m( 0, 2) + p.y * m( 1, 2) + m( 2, 2);
    return point2_t<S>( ( p.x * m( 0, 0) + p.y * m( 1, 0) + m( 2, 0)) / w,
                        ( p.x * m( 0, 1) + p.y * m( 1, 1) + m( 2, 1)) / w);
}

template<class T, class S>
hpoint2_t<S> operator*( const hpoint2_t<S>& p, const matrix33_t<T>& m)
{
    return hpoint2_t<S>( p.x * m( 0, 0) + p.y * m( 1, 0) + p.w * m( 2, 0),
                         p.x * m( 0, 1) + p.y * m( 1, 1) + p.w * m( 2, 1),
                         p.x * m( 0, 2) + p.y * m( 1, 2) + p.w * m( 2, 2));
}

// Adapted from ILM's Imath library.
template<class T>
boost::optional<matrix33_t<T> > invert_gauss_jordan( const matrix33_t<T>& m)
{
    matrix33_t<T> s;
    matrix33_t<T> t( m);

    // Forward elimination
    for( int i = 0; i < 2 ; i++)
    {
        int pivot = i;
        T pivotsize = t( i, i);

        if( pivotsize < 0)
            pivotsize = -pivotsize;

        for( int j = i + 1; j < 3; ++j)
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
            return boost::optional<matrix33_t<T> >();

        if( pivot != i)
        {
            for( int j = 0; j < 3; ++j)
            {
                T tmp = t( i, j);
                t( i, j) = t( pivot, j);
                t( pivot, j) = tmp;
                tmp = s( i, j);
                s( i, j) = s( pivot, j);
                s( pivot, j) = tmp;
            }
        }

        for( int j = i + 1; j < 3; ++j)
        {
            T f = t( j, i) / t( i, i);

            for( int k = 0; k < 3; ++k)
            {
                t( j, k) -= f * t( i, k);
                s( j, k) -= f * s( i, k);
            }
        }
    }

    // Backward substitution
    for( int i = 2; i >= 0; --i)
    {
        T f = t( i, i);

        if( f == 0)
            return boost::optional<matrix33_t<T> >();

        for( int j = 0; j < 3; ++j)
        {
            t( i, j) /= f;
            s( i, j) /= f;
        }

        for( int j = 0; j < i; ++j)
        {
            f = t( j, i);
            for( int k = 0; k < 3; ++k)
            {
                t( j, k) -= f * t( i, k);
                s( j, k) -= f * s( i, k);
            }
        }
    }

    return s;
}

// Adapted from ILM's Imath library.
template<class T>
boost::optional<matrix33_t<T> > invert( const matrix33_t<T>& m)
{
    if( m( 0, 2) != 0 || m( 1, 2) != 0 || m( 2, 2) != 1)
    {
        matrix33_t<T> s(m( 1, 1) * m( 2, 2) - m( 2, 1) * m( 1, 2),
                        m( 2, 1) * m( 0, 2) - m( 0, 1) * m( 2, 2),
                        m( 0, 1) * m( 1, 2) - m( 1, 1) * m( 0, 2),

                        m( 2, 0) * m( 1, 2) - m( 1, 0) * m( 2, 2),
                        m( 0, 0) * m( 2, 2) - m( 2, 0) * m( 0, 2),
                        m( 1, 0) * m( 0, 2) - m( 0, 0) * m( 1, 2),

                        m( 1, 0) * m( 2, 1) - m( 2, 0) * m( 1, 1),
                        m( 2, 0) * m( 0, 1) - m( 0, 0) * m( 2, 1),
                        m( 0, 0) * m( 1, 1) - m( 1, 0) * m( 0, 1));

        T r = m( 0, 0) * s( 0, 0) + m( 0, 1) * s( 1, 0) + m( 0, 2) * s( 2, 0);

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
                        return boost::optional<matrix33_t<T> >();
                }
            }
        }

        return s;
    }
    else
    {
        matrix33_t<T> s( m( 1, 1), -m( 0, 1), 0,
                        -m( 1, 0),  m( 0, 0), 0,
                         0,         0,        1);

        T r = m( 0, 0) * m( 1, 1) - m( 1, 0) * m( 0, 1);

        if( cmath<T>::fabs( r) >= 1)
        {
            for( int i = 0; i < 2; ++i)
            {
                for( int j = 0; j < 2; ++j)
                    s( i, j) /= r;
            }
        }
        else
        {
            T mr = cmath<T>::fabs( r) / std::numeric_limits<T>::min();

            for( int i = 0; i < 2; ++i)
            {
                for( int j = 0; j < 2; ++j)
                {
                    if( mr > cmath<T>::fabs( s( i, j)))
                        s( i, j) /= r;
                    else
                        return boost::optional<matrix33_t<T> >();
                }
            }
        }

        s( 2, 0) = -m( 2, 0) * s( 0, 0) - m( 2, 1) * s( 1, 0);
        s( 2, 1) = -m( 2, 0) * s( 0, 1) - m( 2, 1) * s( 1, 1);
        return s;
    }
}

typedef matrix33_t<float>   matrix33f_t;
typedef matrix33_t<double>  matrix33d_t;

#ifdef RAMEN_WITH_HALF
    typedef matrix33_t<half> matrix33h_t;
#endif

} // math
} // ramen

#endif
