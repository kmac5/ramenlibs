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

#include<boost/assign.hpp>

#include<ramen/arrays/named_array_map.hpp>

using namespace ramen::core;
using namespace ramen::arrays;

#include"test_visitors.hpp"

TEST( NamedArrayMap, Construct)
{
    named_array_map_t map;
    ASSERT_TRUE( map.empty());
    ASSERT_EQ( map.size(), 0);
}

TEST( NamedArrayMap, Insert)
{
    named_array_map_t map;
    ASSERT_TRUE( map.empty());

    map.insert( name_t( "C"), color3f_k);
    ASSERT_EQ( map.size(), 1);

    map.insert( name_t( "Z"), float_k);
    ASSERT_EQ( map.size(), 2);
    
    map.erase( name_t( "Z"));
    ASSERT_EQ( map.size(), 1);
}

TEST( NamedArrayMap, ApplyVisitor)
{
    named_array_map_t array_map;
    
    name_t A_name( "A");
    name_t B_name( "B");
    name_t C_name( "C");
    
    array_map.insert( A_name, float_k);
    {
        array_ref_t<float> xref( array_map.array( A_name));
        boost::assign::push_back( xref) = 1.0f, 1.0f, 3.0f, 7.0f, 8.0f, 11.0f;
    }

    array_map.insert( B_name, uint32_k);
    {
        array_ref_t<boost::uint32_t> yref( array_map.array( B_name));    
        boost::assign::push_back( yref) = 1, 1, 3, 7, 8, 11, 17;
    }
    
    double_visitor v;
    array_map.apply_visitor( v);
    
    array_ref_t<float> xref( array_map.array( A_name));
    EXPECT_FLOAT_EQ( xref[0], 2.0f);
    EXPECT_FLOAT_EQ( xref[2], 6.0f);
    EXPECT_FLOAT_EQ( xref[4], 16.0f);

    array_ref_t<boost::uint32_t> yref( array_map.array( B_name));    
    EXPECT_EQ( yref[0], 2);
    EXPECT_EQ( yref[2], 6);
    EXPECT_EQ( yref[4], 16);
}   

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
