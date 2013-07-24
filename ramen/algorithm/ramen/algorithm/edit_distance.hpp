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

// Based on Levenshtein Distance Algorithm:
// public domain C++ implementation by Anders Sewerin Johansen.

#ifndef RAMEN_ALGORITHM_EDIT_DISTANCE_HPP
#define RAMEN_ALGORITHM_EDIT_DISTANCE_HPP

#include<ramen/algorithm/config.hpp>

#include<cassert>
#include<algorithm>

#include<boost/scoped_array.hpp>

namespace ramen
{
namespace algorithm
{

template<class T>
class edit_distance_t
{
public:

    typedef typename T::value_type char_type;

    edit_distance_t() : rows_( 0), cols_( 0) {}

    int operator()( const T& src, const T& dst)
    {
        int n = src.size();
        int m = dst.size();

        if( n == 0)
          return m;

        if( m == 0)
          return n;

        if( rows_ < n + 1 || cols_ < m + 1)
            realloc_matrix( n + 1, m + 1);

        for( int i = 0; i <= n; i++)
          matrix( i, 0) = i;

        for( int j = 0; j <= m; j++)
          matrix( 0, j) = j;

        for( int i = 1; i <= n; i++)
        {
            char_type s_i = src[i-1];

            for( int j = 1; j <= m; j++)
            {
                char_type t_j = dst[j-1];
                int cost;

                if( s_i == t_j)
                    cost = 0;
                else
                    cost = 1;

                int above = matrix( i - 1, j);
                int left = matrix( i, j - 1);
                int diag = matrix( i - 1, j - 1);
                int cell = std::min( above + 1, std::min( left + 1, diag + cost));

                if( i > 2 && j > 2)
                {
                    int trans = matrix( i - 2, j - 2) + 1;

                    if( src[i-2] != t_j)
                        ++trans;

                    if( s_i != dst[j-2])
                        ++trans;

                    if( cell > trans)
                        cell = trans;
                }

                matrix( i, j) = cell;
            }
        }

        return matrix( n, m);
    }

private:

    void realloc_matrix( int rows, int cols)
    {
        matrix_.reset( new int[rows * cols]);
        rows_ = rows;
        cols_ = cols;
    }

    int matrix( int i, int j) const
    {
        assert( i >= 0 && i < rows_);
        assert( j >= 0 && j < cols_);

        return matrix_[i * cols_ + j];
    }

    int& matrix( int i, int j)
    {
        assert( i >= 0 && i < rows_);
        assert( j >= 0 && j < cols_);

        return matrix_[i * cols_ + j];
    }

    boost::scoped_array<int> matrix_;
    int rows_, cols_;
};

} // algorithm
} // ramen

#endif
