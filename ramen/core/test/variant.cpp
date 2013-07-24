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

#include<ramen/core/variant.hpp>

using namespace ramen::core;

TEST( Variant, Construct)
{
    variant_t x( 77.0f);
    ASSERT_EQ( x.type(), float_k);
    ASSERT_EQ( get<float>( x), 77.0f);

    variant_t y( 11);
    ASSERT_EQ( y.type(), int32_k);
    ASSERT_EQ( get<boost::int32_t>( y), 11);

    boost::uint32_t zval( 33);
    variant_t z( zval);
    ASSERT_EQ( z.type(), uint32_k);
    ASSERT_EQ( get<boost::uint32_t>( z), zval);

    // All lines bellow should generate a compiler error.
    // Uncomment to check...

    //float *ptr_to_float = 0;
    //variant_t variant_with_pointer( ptr_to_float);

    //struct my_class {};
    //my_class my_c;
    //variant_t variant_with_value( my_c);

    //int (*my_fun_ptr)();
    //variant_t variant_with_fun_ptr( my_fun_ptr);
}

TEST( Variant, CopyAssign)
{
    variant_t x( 11);
    variant_t y( x);
    variant_t z;

    ASSERT_EQ( x.type(), y.type());
    ASSERT_EQ( x, y);

    z = y;
    ASSERT_EQ( z.type(), y.type());
    ASSERT_EQ( z, x);
}

TEST( Variant, Get)
{
    variant_t x( 21.0f);
    EXPECT_EQ( x.type(), float_k);
    EXPECT_EQ( get<float>( x), 21.0f);

    EXPECT_EQ( *get<float>( &x), 21.0f);
    EXPECT_FALSE( get<string8_t>( &x));

    get<float>( x) = 1.0f;
    EXPECT_EQ( x.type(), float_k);
    EXPECT_EQ( get<float>( x), 1.0f);

    EXPECT_THROW( get<string8_t>( x), bad_type_cast);
}

TEST( Variant, Equality)
{
    variant_t x( 7.0f);
    variant_t y( 7.0f);

    EXPECT_EQ( x, 7.0f);
    EXPECT_EQ( y, 7.0f);
    EXPECT_EQ( 7.0f, x);
    EXPECT_EQ( x, y);

    variant_t z1( string8_t( "a string"));
    variant_t z2( "a string");
    EXPECT_EQ( z1, z2);

    EXPECT_EQ( get<string8_t>( z1), string8_t( "a string"));
    EXPECT_EQ( z1, string8_t( "a string"));
    EXPECT_EQ( string8_t( "a string"),  z1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
