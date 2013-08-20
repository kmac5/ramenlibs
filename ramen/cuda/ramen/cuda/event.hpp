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

#ifndef RAMEN_CUDA_EVENT_HPP
#define RAMEN_CUDA_EVENT_HPP

#include<ramen/cuda/exceptions.hpp>

namespace ramen
{
namespace cuda
{

class RAMEN_CUDA_API event_t
{
public:

    event_t();
    ~event_t();

    void record();
    void syncronize();

    cudaEvent_t event() const
    {
        return event_;
    }

private:

    // non-copyable
    event_t( const event_t&);
    event_t& operator=( const event_t&);

    cudaEvent_t event_;
};

RAMEN_CUDA_API float elapsed_time( const event_t& start, const event_t& end);

} // cuda
} // ramen

#endif
