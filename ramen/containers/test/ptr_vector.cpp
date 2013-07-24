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

#include<gtest/gtest.h>

#include<ramen/containers/ptr_vector.hpp>

#include<typeinfo>
#include<iostream>

using namespace ramen::core;
using namespace ramen::containers;

class PtrVectorTest : public ::testing::Test {};

TEST_F( PtrVectorTest, All)
{
    ptr_vector_t<int> ivec;
    ASSERT_TRUE( ivec.empty());

    int *i1 = new int( 7);
    int *i2 = new int( 11);
    int *i3 = new int( 17);
    int *i0 = new int( 77);

    ivec.push_back( auto_ptr_t<int>( i1));
    ASSERT_EQ( ivec.size(), 1);
    ASSERT_TRUE( ivec.contains_ptr( i1));

    ivec.push_back( auto_ptr_t<int>( i2));
    ASSERT_EQ( ivec.size(), 2);
    ASSERT_TRUE( ivec.contains_ptr( i2));

    ivec.push_back( auto_ptr_t<int>( i3));
    ASSERT_EQ( ivec.size(), 3);
    ASSERT_TRUE( ivec.contains_ptr( i3));

    auto_ptr_t<int> pi1 = ivec.release_ptr( i1);
    ASSERT_EQ( ivec.size(), 2);
    ASSERT_EQ( *pi1, *i1);

    auto_ptr_t<int> pnull = ivec.release_ptr( i0);
    ASSERT_FALSE( pnull.get());

    delete i0;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
