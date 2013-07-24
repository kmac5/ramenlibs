/**********************************************************************
 * Copyright (C) 2013 Esteban Tovagliari. All Rights Reserved.        *
 **********************************************************************/

#include<gtest/gtest.h>

#include<ramen/algorithm/for_each_position.hpp>

using namespace ramen::algorithm;

struct for_each_pos_test1_fun
{
    for_each_pos_test1_fun()
    {
        num_times = 0;
    }

    void operator()( float *it)
    {
        *it = num_times++;
    }

private:

    float *last_iter;
    int num_times;
};

TEST( ForEachPosition, All)
{
    int float_lenght = 20;
    float *float_array = new float[float_lenght];
    for_each_pos_test1_fun f;
    for_each_position( float_array, float_array + float_lenght, f);
    for( int i = 0; i < float_lenght; ++i)
        EXPECT_EQ( float_array[i], i);

    delete [] float_array;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
