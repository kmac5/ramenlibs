/**********************************************************************
 * Copyright (C) 2013 Esteban Tovagliari. All Rights Reserved.        *
 **********************************************************************/

#include<gtest/gtest.h>

#include<ramen/algorithm/clamp.hpp>

using namespace ramen::algorithm;

TEST( Clamp, All)
{
    ASSERT_EQ( clamp( 7, 0, 11), 7);
    ASSERT_EQ( clamp( -2, 0, 11), 0);
    ASSERT_EQ( clamp( 77, 0, 11), 11);
    ASSERT_EQ( clamp( 3, 3, 3), 3);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
