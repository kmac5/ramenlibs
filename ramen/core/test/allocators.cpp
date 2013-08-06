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

#include<gmock/gmock.h>
#include<gtest/gtest.h>

#include<ramen/core/allocator_interface.hpp>
#include<ramen/core/new_allocator.hpp>
#include<ramen/core/stl_allocator_adapter.hpp>

#include<boost/shared_ptr.hpp>
#include<boost/container/vector.hpp>

#include<iostream>

using namespace ramen::core;

TEST( Allocators, GlobalNewAllocator)
{
    allocator_ptr_t alloc( global_new_allocator());
    allocator_ptr_t alloc2( global_new_allocator());
    ASSERT_EQ( alloc, alloc2);
}

TEST( Allocators, All)
{
    allocator_ptr_t alloc( global_new_allocator());
    stl_allocator_adapter_t<int> alloc_wrapper( alloc);
    typedef boost::container::vector<int, stl_allocator_adapter_t<int> > vector_type;

    vector_type vec( alloc_wrapper);
    boost::container::vector<int> vec_no_alloc;

    for( int i = 0; i < 100; ++i)
    {
        vec.push_back( i);
        vec_no_alloc.push_back( i);
    }

    EXPECT_EQ( vec.size(), vec_no_alloc.size());
    EXPECT_EQ( vec.capacity(), vec_no_alloc.capacity());

    vector_type vec2 = vec;
    vec.swap( vec2);

    vec.clear();
    vec.reserve( 100);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleMock( &argc, argv);
    return RUN_ALL_TESTS();
}
