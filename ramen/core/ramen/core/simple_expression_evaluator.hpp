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

#ifndef RAMEN_SIMPLE_EXPRESSION_EVALUATOR_HPP
#define RAMEN_SIMPLE_EXPRESSION_EVALUATOR_HPP

#include<ramen/core/config.hpp>

#include<ramen/core/string8.hpp>

#include<boost/optional.hpp>

namespace ramen
{
namespace core
{

class RAMEN_CORE_API simple_expression_evaluator_t
{
public:

    simple_expression_evaluator_t();
    ~simple_expression_evaluator_t();

    boost::optional<double> operator()( const core::string8_t& str) const;

private:

    // non-copyable
    simple_expression_evaluator_t( const simple_expression_evaluator_t&);
    simple_expression_evaluator_t& operator=( const simple_expression_evaluator_t&);

    struct impl;
    impl *pimpl_;
};

} // core
} // ramen

#endif
