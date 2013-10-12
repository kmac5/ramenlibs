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

#include<ramen/arrays/named_array_map.hpp>

using namespace ramen::core;
using namespace ramen::arrays;

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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
