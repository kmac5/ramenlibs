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

#include<ramen/functional/functional.hpp>

using namespace ramen;

TEST( Functional, Basic)
{
    functional::plus<int> fplus;
    EXPECT_EQ( fplus( 7, 4), 11);

    functional::minus<int> fminus;
    EXPECT_EQ( fminus( 7, 4), 3);
}

TEST( Functional, MultPlus)
{
    functional::multiply_plus<int,int> fmplus;
    EXPECT_EQ( fmplus( 7, 4, 3), 19);
}

TEST( Functional, OpAssign)
{
    int x = 0;
    functional::plus_assign<int> fplus;
    fplus( x, 4);
    EXPECT_EQ( x, 4);
    fplus( x, 7);
    EXPECT_EQ( x, 11);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
