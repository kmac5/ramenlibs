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

//#include<ramen/core/string8.hpp>
#include<ramen/containers/string8_vector.hpp>

using namespace ramen::core;
using namespace ramen::containers;

TEST( String8Vector, Construct)
{
    string8_vector_t vec;
    ASSERT_TRUE( vec.empty());
    ASSERT_EQ( vec.size(), 0);

    vec.push_back( string8_t( "The quick brown fox"));
    ASSERT_FALSE( vec.empty());
    ASSERT_EQ( vec.size(), 1);
}

TEST( String8Vector, Equality)
{
    string8_vector_t vec;
    vec.push_back( string8_t( "The quick brown fox"));
    vec.push_back( string8_t( "jumps over the lazy dog"));

    string8_vector_t vec2;
    vec2.push_back( string8_t( "The quick brown fox"));
    vec2.push_back( string8_t( "jumps over the lazy dog"));

    ASSERT_EQ( vec, vec2);
}

TEST( String8Vector, CopyAssign)
{
    string8_vector_t vec;
    vec.push_back( string8_t( "The quick brown fox"));
    vec.push_back( string8_t( "jumps over the lazy dog"));

    string8_vector_t vec2( vec);
    ASSERT_EQ( vec, vec2);
}

TEST( String8Vector, Swap)
{
}

TEST( String8Vector, ConstIndexing)
{
    string8_vector_t vec;
    vec.push_back( string8_t( "The quick brown fox"));
    vec.push_back( string8_t( "jumps over the lazy dog"));
    ASSERT_FALSE( vec.empty());
    ASSERT_EQ( vec.size(), 2);
    ASSERT_EQ( vec[0], string8_t( "The quick brown fox"));
    ASSERT_EQ( vec[1], string8_t( "jumps over the lazy dog"));
}

TEST( String8Vector, NonConstIndexing)
{
    string8_vector_t vec;
    vec.push_back( string8_t( "The quick brown fox"));
    vec.push_back( string8_t( "jumps over the lazy dog"));

    vec[0] = string8_t( "The quick white cat");
    ASSERT_EQ( vec[0], string8_t( "The quick white cat"));
    ASSERT_EQ( vec[1], string8_t( "jumps over the lazy dog"));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
