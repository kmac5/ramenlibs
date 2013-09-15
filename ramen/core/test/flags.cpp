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

#include<ramen/core/flags.hpp>

using namespace ramen::core;

enum flag_bits_t
{
    bit0 = 1 << 0,
    bit1 = 1 << 1,
    bit2 = 1 << 2
};

TEST( Flags, All)
{
    int flags = 0;

    set_flag( flags, bit0, true);
    ASSERT_TRUE( test_flag( flags, bit0));

    set_flag( flags, bit2, true);
    ASSERT_TRUE( test_flag( flags, bit2));

    set_flag( flags, bit0, false);
    ASSERT_FALSE( test_flag( flags, bit0));

    set_flag( flags, bit2, false);
    ASSERT_FALSE( test_flag( flags, bit2));

    ASSERT_EQ( flags, 0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
