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

#ifndef RAMEN_MATH_TRANSFORM44_STACK_HPP
#define RAMEN_MATH_TRANSFORM44_STACK_HPP

#include<ramen/math/transform44.hpp>

#include<vector>

namespace ramen
{
namespace math
{

template<class T>
class transform44_stack_t
{
public:

    transform44_stack_t()
    {
        reset();
    }

    const transform44_t<T>& top() const
    {
        assert( !stack_.empty());

        return stack_.back();
    }

    void push()
    {
        stack_.push_back( top());
    }

    void pop()
    {
        assert( stack_.size() > 1);

        stack_.pop_back();
    }

    void set_transform( const transform44_t<T>& t)
    {
        stack_.back() = t;
    }

    void concat_transform( const transform44_t<T>& t)
    {
        stack_.back() *= t;
    }

    void reset()
    {
        stack_.clear();
        stack_.push_back( transform44_t<T>());
    }

private:

    std::vector<transform44_t<T> > stack_;
};

} // math
} // ramen

#endif
