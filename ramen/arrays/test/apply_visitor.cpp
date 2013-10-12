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

#include<boost/assign.hpp>

#include<ramen/arrays/apply_visitor.hpp>

using namespace ramen::core;
using namespace ramen::arrays;

#include"test_visitors.hpp"

TEST( ApplyVisitor, Const)
{
    max_visitor v;

    array_t x( float_k);
    array_ref_t<float> xref( x);
    boost::assign::push_back( xref) = 1.0f, 1.0f, 3.0f, 7.0f, 8.0f, 11.0f;
    
    const array_t& xx( x);    
    apply_visitor( v, xx);
    EXPECT_DOUBLE_EQ( v.max_value, 11.0);
    
    array_t y( uint32_k);
    array_ref_t<boost::uint32_t> yref( y);
    boost::assign::push_back( yref) = 1, 1, 3, 7, 8, 11, 17;

    const array_t& yy( y);    
    apply_visitor( v, yy);
    EXPECT_DOUBLE_EQ( v.max_value, 17.0);
}

TEST( ApplyVisitor, NonConst)
{
    double_visitor v;

    array_t x( float_k);
    array_ref_t<float> xref( x);
    boost::assign::push_back( xref) = 1.0f, 1.0f, 3.0f, 7.0f, 8.0f, 11.0f;
    apply_visitor( v, x);
    EXPECT_FLOAT_EQ( xref[0], 2.0f);
    EXPECT_FLOAT_EQ( xref[2], 6.0f);
    EXPECT_FLOAT_EQ( xref[4], 16.0f);

    array_t y( uint32_k);
    array_ref_t<boost::uint32_t> yref( y);
    boost::assign::push_back( yref) = 1, 1, 3, 7, 8, 11;
    apply_visitor( v, y);
    EXPECT_EQ( yref[0], 2);
    EXPECT_EQ( yref[2], 6);
    EXPECT_EQ( yref[4], 16);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
