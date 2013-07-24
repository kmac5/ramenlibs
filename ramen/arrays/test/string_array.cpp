/**********************************************************************
 * Copyright (C) 2013 Esteban Tovagliari. All Rights Reserved.        *
 **********************************************************************/

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
