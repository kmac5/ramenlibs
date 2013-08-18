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

#ifndef RAMEN_FUNCTIONAL_FUNCTIONAL_HPP
#define RAMEN_FUNCTIONAL_FUNCTIONAL_HPP

#include<ramen/functional/config.hpp>

#include<functional>

namespace ramen
{
namespace functional
{

template<class T>
struct additive_identity
{
    static T value() { return T( 0);}
};

template<class T>
struct multiplicative_identity
{
    static T value() { return T( 1);}
};

template<class Arg1, class Arg2, class Arg3, class Result>
struct ternary_function
{
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Arg3 third_argument_type;
    typedef Result result_type;
};

template<class T>
struct plus : public std::binary_function<T,T,T>
{
    T operator()( const T& x, const T& y) const
    {
        return x + y;
    }
};

template<class T>
struct minus : public std::binary_function<T,T,T>
{
    T operator()( const T& x, const T& y) const
    {
        return x - y;
    }
};

template<class T, class S>
struct multiply : public std::binary_function<T,S,T>
{
    T operator()( const T& x, const S& y) const
    {
        return x * y;
    }
};

template<class T, class S>
struct divide : public std::binary_function<T,S,T>
{
    T operator()( const T& x, const S& y) const
    {
        return x / y;
    }
};

template<class T, class S>
struct multiply_plus : public ternary_function<T,T,S,T>
{
    T operator()( const T& x, const T& y, const S& s) const
    {
        return plus<T>()( x, multiply<T,S>()( y, s));
    }
};

template<class T>
struct plus_assign : public std::binary_function<T,T,void>
{
    void operator()( T& x, const T& y) const
    {
        x += y;
    }
};

template<class T>
struct minus_assign : public std::binary_function<T,T,void>
{
    void operator()( T& x, const T& y) const
    {
        x -= y;
    }
};

template<class T, class S>
struct multiply_assign : public std::binary_function<T,S,void>
{
    void operator()( T& x, const S& y) const
    {
        x *= y;
    }
};

template<class T, class S>
struct divide_assign : public std::binary_function<T,T,void>
{
    void operator()( T& x, const S& y) const
    {
        x /= y;
    }
};

template<class T, class S>
struct multiply_plus_assign : public ternary_function<T,T,S,T>
{
    void operator()( T& x, const T& y, const S& s) const
    {
        plus_assign<T>()( x, multiply<T,S>()( y, s));
    }
};

template<class T>
struct negate : public std::unary_function<T,T>
{
    T operator()( const T& x) const
    {
        return -x;
    }
};

} // functional
} // ramen

#endif
