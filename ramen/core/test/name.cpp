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

#include<ramen/core/name.hpp>

using namespace ramen::core;

name_t global_name( "global_name");

TEST( Name, Basic)
{
    name_t a;
    name_t b( "");
    ASSERT_EQ( a.c_str(), b.c_str());
    EXPECT_TRUE( a.empty());
    EXPECT_TRUE( b.empty());

    name_t p( "P");
    name_t q( "Q");
    ASSERT_NE( p.c_str(), q.c_str());

    name_t p2( "P");
    ASSERT_EQ( p.c_str(), p2.c_str());

    swap( p, q);
    ASSERT_EQ( q.c_str(), p2.c_str());

    name_t X( "");
    ASSERT_EQ( a.c_str(), X.c_str());
    ASSERT_EQ( b.c_str(), X.c_str());

    name_t global_name2( "global_name");
    ASSERT_EQ( global_name, global_name2);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
