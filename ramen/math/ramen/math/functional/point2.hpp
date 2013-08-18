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

#ifndef RAMEN_MATH_FUNCTIONAL_POINT2_HPP
#define RAMEN_MATH_FUNCTIONAL_POINT2_HPP

#include<ramen/math/point2.hpp>

#include<ramen/functional/functional.hpp>

namespace ramen
{
namespace functional
{

template<class T>
struct plus<math::point2_t<T> > : public std::binary_function<math::point2_t<T>,
                                                              math::point2_t<T>,
                                                              math::point2_t<T> >
{
    math::point2_t<T> operator()( const math::point2_t<T>& a, const math::point2_t<T>& b) const
    {
        return math::point2_t<T>( a.x + b.x, a.y + b.y);
    }
};

template<class T>
struct minus<math::point2_t<T> > : public std::binary_function<math::point2_t<T>,
                                                               math::point2_t<T>,
                                                               math::point2_t<T> >
{
    math::point2_t<T> operator()( const math::point2_t<T>& a, const math::point2_t<T>& b) const
    {
        return math::point2_t<T>( a.x - b.x, a.y - b.y);
    }
};

template<class T, class S>
struct multiply<math::point2_t<T>, S> : public std::binary_function<math::point2_t<T>,
                                                                    S,
                                                                    math::point2_t<T> >
{
    math::point2_t<T> operator()( const math::point2_t<T>& a, const S& b) const
    {
        return math::point2_t<T>( a.x * b, a.y * b);
    }
};

template<class T, class S>
struct divide<math::point2_t<T>, S> : public std::binary_function<math::point2_t<T>,
                                                                  S,
                                                                  math::point2_t<T> >
{
    math::point2_t<T> operator()( const math::point2_t<T>& a, const S& b) const
    {
        return math::point2_t<T>( a.x / b, a.y / b);
    }
};

template<class T, class S>
struct multiply_plus<math::point2_t<T>, S> : public ternary_function<math::point2_t<T>,
                                                                     math::point2_t<T>,
                                                                     S,
                                                                     math::point2_t<T> >
{
    math::point2_t<T> operator()( const math::point2_t<T>& a, const math::point2_t<T>& b, const S& s) const
    {
        return math::point2_t<T>( a.x + s * b.x,
                                  a.y + s * b.y);
    }
};

template<class T>
struct plus_assign<math::point2_t<T> > : public std::binary_function<math::point2_t<T>,
                                                                     math::point2_t<T>,
                                                                     void>
{
    void operator()( math::point2_t<T>& a, const math::point2_t<T>& b) const
    {
        a.x += b.x;
        a.y += b.y;
    }
};

template<class T>
struct minus_assign<math::point2_t<T> > : public std::binary_function<math::point2_t<T>,
                                                                      math::point2_t<T>,
                                                                      void>
{
    void operator()( math::point2_t<T>& a, const math::point2_t<T>& b) const
    {
        a.x -= b.x;
        a.y -= b.y;
    }
};

template<class T, class S>
struct multiply_assign<math::point2_t<T>, S> : public std::binary_function<math::point2_t<T>,
                                                                           S,
                                                                           void>
{
    void operator()( math::point2_t<T>& a, const S& b) const
    {
        a.x *= b;
        a.y *= b;
    }
};

template<class T, class S>
struct divide_assign<math::point2_t<T>, S> : public std::binary_function<math::point2_t<T>,
                                                                         S,
                                                                         void>
{
    void operator()( math::point2_t<T>& a, const S& b) const
    {
        a.x /= b;
        a.y /= b;
    }
};

template<class T, class S>
struct multiply_plus_assign<math::point2_t<T>, S> : public ternary_function<math::point2_t<T>,
                                                                            math::point2_t<T>,
                                                                            S,
                                                                            math::point2_t<T> >
{
    void operator()( math::point2_t<T>& a, const math::point2_t<T>& b, const S& s) const
    {
        a.x += s * b.x;
        a.y += s * b.y;
    }
};

} // functional
} // ramen

#endif
