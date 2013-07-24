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

#include<cstring>
#include<iostream>

#include<ramen/core/string8.hpp>

#include<boost/range/algorithm/for_each.hpp>
#include<boost/algorithm/string/find.hpp>
#include<boost/tokenizer.hpp>
#include<boost/lexical_cast.hpp>

using namespace ramen::core;

TEST( String8, Construct)
{
    string8_t a( "XYZ");
    ASSERT_EQ( a.size(), 3);
    ASSERT_EQ( strcmp( a.c_str(), "XYZ"), 0);

    string8_t b( "The quick brown fox jumps over the lazy dog");
    string8_t part0( b, 0, 9);
    ASSERT_EQ( part0, "The quick");

    string8_t part1( b, 10, 5);
    ASSERT_EQ( part1, "brown");

    string8_t part2( b, b.length() - 3, 150);
    ASSERT_EQ( part2, "dog");
}

TEST( String8, CopyAssign)
{
    string8_t a( "XYZ");
    string8_t b( a);

    ASSERT_EQ( a.size(), b.size());
    ASSERT_EQ( strcmp( a.c_str(), b.c_str()), 0);
    ASSERT_EQ( strcmp( a.c_str(), "XYZ"), 0);
    ASSERT_NE( a.c_str(), b.c_str());
}

TEST( String8, MoveAssign)
{
    string8_t a( "XYZ");
    string8_t b( "XYZ");

    string8_t c( boost::move( b));
    EXPECT_EQ( a.size(), 3);
    EXPECT_EQ( c, a);

    b = boost::move( c);
    EXPECT_EQ( b.size(), 3);
    EXPECT_EQ( b, a);
}

TEST( String8, Swap)
{
    string8_t sp( "Structured procrastination");
    string8_t wa( "Wasabi alarm");

    sp.swap( wa);

    EXPECT_EQ( strcmp( sp.c_str(), "Wasabi alarm"), 0);
    EXPECT_EQ( strcmp( wa.c_str(), "Structured procrastination"), 0);
}

TEST( String8, Compare)
{
    string8_t sp( "Structured procrastination");
    string8_t sp2( "Structured procrastination");

    string8_t wa( "Wasabi alarm");
    string8_t wa2( "Wasabi alarm");

    EXPECT_EQ( strcmp( wa.c_str(), "Wasabi alarm"), 0);
    EXPECT_EQ( wa, wa2);
    EXPECT_NE( wa, sp2);

    EXPECT_EQ( strcmp( sp.c_str(), "Structured procrastination"), 0);
    EXPECT_EQ( sp, sp2);
    EXPECT_NE( sp, wa);
}

TEST( String8, Concat)
{
    string8_t a( "XXX");
    a += "YYY";
    a += string8_t( "ZZZ");

    EXPECT_EQ( a.size(), 9);
    EXPECT_EQ( strcmp( a.c_str(), "XXXYYYZZZ"), 0);
}

TEST( String8, Iterators)
{
    string8_t a( "XYZ");
    string8_t::const_iterator it( a.begin());
    string8_t::const_iterator e( a.end());

    EXPECT_NE( it, e);
    EXPECT_EQ( std::distance( it, e), a.size());

    EXPECT_EQ( it[0], 'X');
    EXPECT_EQ( it[1], 'Y');
    EXPECT_EQ( it[2], 'Z');

    string8_t b;
    EXPECT_EQ( b.begin(), b.end());
}

TEST( String8, MakeString)
{
    string8_t str( make_string( "Xyzw ", "Yzw ", "Zw ", "W"));
    EXPECT_EQ( str, "Xyzw Yzw Zw W");

    str = make_string( "1234 ", "567 ", "89 ");
    EXPECT_EQ( str, "1234 567 89 ");

    str = make_string( string8_t("1234").c_str(), "567");
    EXPECT_EQ( str, "1234567");
}

TEST( String8, OperatorPlus)
{
    string8_t a( "XYZ");
    a = a + string8_t( "ABC");
    a = a + "DEF";

    EXPECT_EQ( a.size(), 9);
    EXPECT_EQ( strcmp( a.c_str(), "XYZABCDEF"), 0);
}

struct nop_fun
{
    template<class T>
    void operator()( T x)
    {
    }
};

TEST( String8, BoostRange)
{
    string8_t a( "XYZ");
    boost::range::for_each( a, nop_fun());
}

TEST( String8, BoostStringAlgo)
{
    string8_t str( make_string( "Xyzw ", "Yzw ", "Zw ", "W"));
    boost::iterator_range<string8_t::iterator> range = boost::algorithm::find_last( str, "Z");
    EXPECT_EQ( *range.begin(), 'Z');
    EXPECT_EQ( std::distance( str.begin(), range.begin()), 9);
}

TEST( String8, STLString)
{
    std::string stest( "My quick quasi constant 7312.");
    string8_t dtest( stest);
    EXPECT_EQ( stest.size(), dtest.size());
    EXPECT_EQ( strcmp( stest.c_str(), dtest.c_str()), 0);

    std::string xtest( dtest.to_std_string());
    EXPECT_EQ( stest, xtest);
}

TEST( String8, BoostTokenizer)
{
    string8_t test_path( "/usr/local/bin/any_bin");
    boost::char_separator<string8_t::char_type> sep( "/");

    typedef boost::tokenizer<boost::char_separator<char>,
                             string8_t::const_iterator,
                             string8_t> tokenizer_type;

    tokenizer_type tokens( test_path, sep);

    tokenizer_type::iterator it( tokens.begin());
    EXPECT_EQ( *it, "usr");
    ++it;
    EXPECT_EQ( *it, "local");
    ++it;
    EXPECT_EQ( *it, "bin");
    ++it;
    EXPECT_EQ( *it, "any_bin");
}

TEST( String8, BoostLexicalCast)
{
    string8_t str( "751");
    int x = boost::lexical_cast<int>( str);
    EXPECT_EQ( x, 751);

    string8_t str2 = boost::lexical_cast<string8_t>( x);
    EXPECT_EQ( str2, str);

    string8_t str3( "any_string_123");
    EXPECT_ANY_THROW( boost::lexical_cast<int>( str3));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
