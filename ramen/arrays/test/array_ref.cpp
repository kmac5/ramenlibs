/**********************************************************************
 * Copyright (C) 2013 Esteban Tovagliari. All Rights Reserved.        *
 **********************************************************************/

#include<gtest/gtest.h>

#include<ramen/arrays/array_ref.hpp>

#include<boost/range/algorithm/for_each.hpp>
#include<boost/range/algorithm/copy.hpp>

#include<boost/assign.hpp>

using namespace ramen::core;
using namespace ramen::arrays;

namespace
{

struct inspect_fun
{
    void operator()( int x) const {}
};

struct mutate_fun
{
    void operator()( int& x) const { ++x;}
};

} // unnamed

TEST( ArrayRef, Construct)
{
}

TEST( ArrayRef, STLAlgo)
{
    const int size = 77;
    array_t x( int32_k);
    x.reserve( size);
    ASSERT_EQ( x.empty(), true);
    ASSERT_EQ( x.capacity(), size);
    ASSERT_EQ( x.type(), int32_k);

    {
        array_ref_t<boost::int32_t> xref( x);

        for( int i = 0; i < size; ++i)
            xref.push_back( i);

        EXPECT_EQ( x.empty(), false);
        EXPECT_EQ( x.size(), size);
        EXPECT_EQ( x.size(), xref.size());
    }

    {
        const_array_ref_t<boost::int32_t> cxref( x);
        boost::range::for_each( cxref, inspect_fun());
    }

    {
        const_array_ref_t<boost::int32_t> xref( x);
        boost::range::for_each( xref, inspect_fun());
    }

    {
        array_ref_t<boost::int32_t> xref( x);
        boost::int32_t k = xref[0];
        boost::range::for_each( xref, mutate_fun());
        EXPECT_EQ( k + 1, xref[0]);
    }

    {
        const_array_ref_t<boost::int32_t> xref( x);

        array_t y( x.type());
        array_ref_t<boost::int32_t> yref( y);

        boost::range::copy( xref, std::back_inserter( yref));
        EXPECT_TRUE( x == y);

        boost::range::copy( xref, yref.begin());
        EXPECT_TRUE( x == y);
    }
}

TEST( ArrayRef, BoostAssign)
{
    array_t x( float_k);
    array_ref_t<float> xref( x);

    boost::assign::push_back( xref) = 1.0f, 1.0f, 3.0f, 7.0f, 8.0f, 11.0f;
    EXPECT_EQ( xref.size(), 6);
    EXPECT_EQ( xref[0], 1.0f);
    EXPECT_EQ( xref[2], 3.0f);
    EXPECT_EQ( xref[3], 7.0f);
    EXPECT_EQ( xref[5], 11.0f);

    boost::assign::push_back( xref) = 1.0f, 1.0f, 3.0f, 7.0f, 8.0f, 11.0f;
    EXPECT_EQ( xref.size(), 12);
    EXPECT_EQ( xref[6], 1.0f);
    EXPECT_EQ( xref[8], 3.0f);
    EXPECT_EQ( xref[9], 7.0f);
    EXPECT_EQ( xref[11], 11.0f);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
