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

#include<ramen/cuda/auto_ptr.hpp>

#include<boost/utility/identity_type.hpp>

using namespace ramen::cuda;

void sink_function( BOOST_RV_REF( BOOST_IDENTITY_TYPE(( auto_ptr_t<int, device_ptr_policy>))) x)
{
    auto_ptr_t<int,device_ptr_policy> z( x);
}

auto_ptr_t<int,device_ptr_policy> source_function()
{
    return auto_ptr_t<int,device_ptr_policy>( cuda_malloc<int>( 1));
}

TEST( AutoPtr, Construct)
{
    auto_ptr_t<int,device_ptr_policy> x;
    ASSERT_FALSE( x.get());

    auto_ptr_t<int,device_ptr_policy> y( cuda_malloc<int>( 1));
    ASSERT_TRUE( y);

    x.reset( y.release());
    ASSERT_TRUE( x);
    ASSERT_FALSE( y.get());
}

TEST( AutoPtr, Move)
{
    auto_ptr_t<int,device_ptr_policy> x;
    EXPECT_FALSE( x.get());

    auto_ptr_t<int,device_ptr_policy> y( cuda_malloc<int>( 1));
    EXPECT_TRUE( y);

    x = boost::move( y);
    EXPECT_TRUE( x);
    EXPECT_FALSE( y.get());

    auto_ptr_t<int,device_ptr_policy> z( boost::move( x));
    EXPECT_TRUE( z);
    EXPECT_FALSE( x.get());
}

TEST( AutoPtr, SinkSource)
{
    auto_ptr_t<int,device_ptr_policy> y( cuda_malloc<int>( 1));
    EXPECT_TRUE( y);

    sink_function( boost::move( y));
    EXPECT_FALSE( y.get());

    auto_ptr_t<int,device_ptr_policy> yy = source_function();
    EXPECT_TRUE( yy);
}

TEST( AutoPtr, Container)
{
    /*
    boost::container::vector<auto_ptr_t<int> > vec;
    auto_ptr_t<int> y( new int( 11));

    vec.push_back( boost::move( y));
    EXPECT_FALSE( y.get());
    EXPECT_EQ( vec.size(), 1);

    auto_ptr_t<int> z( new int( 1));
    vec.push_back( boost::move( z));
    EXPECT_FALSE( z.get());
    EXPECT_EQ( vec.size(), 2);

    EXPECT_EQ( *vec[0], 11);
    EXPECT_EQ( *vec[1], 1);
    */
}

TEST( AutoPtr, Host)
{
    auto_ptr_t<int, host_ptr_policy> x;
    EXPECT_FALSE( x.get());

    auto_ptr_t<int, host_ptr_policy> y( cuda_host_alloc<int>( 1));
    EXPECT_TRUE( y);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
