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

#ifndef RAMEN_CUDA_EXCEPTIONS_HPP
#define RAMEN_CUDA_EXCEPTIONS_HPP

#include<ramen/cuda/config.hpp>

#include<ramen/core/exceptions.hpp>

#include<cuda.h>
#include<cuda_runtime_api.h>

namespace ramen
{
namespace cuda
{

class RAMEN_CUDA_API exception : public core::exception
{
public:

    explicit exception();
};

class RAMEN_CUDA_API runtime_error : public exception
{
public:

    explicit runtime_error( cudaError_t err);

    virtual const char* what() const throw();

    cudaError_t cuda_error() const { return error_;}

protected:

    cudaError_t error_;
};

} // cuda
} // ramen

#endif
