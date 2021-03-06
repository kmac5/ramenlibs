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

#include<ramen/arrays/array.hpp>
#include<ramen/arrays/array_ref.hpp>

using namespace ramen::core;
using namespace ramen::arrays;

TEST( Array, StringArray)
{
    array_t x( string8_k);
    array_ref_t<string8_t> x_ref( x);
    x_ref.push_back( string8_t( "xxx"));
    x_ref.push_back( string8_t( "yyy"));
    EXPECT_EQ( x.size(), 2);
    EXPECT_EQ( x_ref.size(), x.size());

    EXPECT_EQ( x_ref[0], string8_t( "xxx"));
    EXPECT_EQ( x_ref[1], string8_t( "yyy"));

    EXPECT_EQ( std::distance( x_ref.begin(), x_ref.end()), 3);

    //for( array_ref_t<string8_t>::iterator it( x_ref.begin()), e( x_ref.end()); it != e; ++it)
    //    ;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
