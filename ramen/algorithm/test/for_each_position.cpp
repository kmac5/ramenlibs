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

#include<boost/scoped_array.hpp>

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
    boost::scoped_array<float> float_array( new float[float_lenght]);
    for_each_pos_test1_fun f;
    for_each_position( float_array.get(), float_array.get() + float_lenght, f);
    for( int i = 0; i < float_lenght; ++i)
        EXPECT_EQ( float_array[i], i);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
