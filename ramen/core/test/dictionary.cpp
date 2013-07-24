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

#include<map>

#include<ramen/core/dictionary.hpp>

#include<boost/foreach.hpp>

#include<ramen/core/string8.hpp>

using namespace ramen::core;
using namespace ramen::math;

TEST( Dictionary, Construct)
{
    dictionary_t dict;

    ASSERT_TRUE( dict.empty());
    ASSERT_TRUE( dict.size() == 0);
}

TEST( Dictionary, CopyAssign)
{
    dictionary_t dict;
    dict.insert()( dictionary_t::key_type( "key1"), dictionary_t::value_type( 7.0f))
                 ( "key2", dictionary_t::value_type( string8_t( "xxxyyy")))
                 ( "key3", dictionary_t::value_type( string8_t( "another_string")));

    dictionary_t dict2( dict);
    EXPECT_TRUE( dict == dict2);

    dictionary_t dict3;
    dict3 = dict;
    EXPECT_TRUE( dict == dict3);
}

TEST( Dictionary, MoveAssign)
{
    dictionary_t dict;
    dict.insert()( dictionary_t::key_type( "key1"), dictionary_t::value_type( 7.0f))
                 ( "key2", dictionary_t::value_type( string8_t( "xxxyyy")))
                 ( "key3", dictionary_t::value_type( string8_t( "another_string")));

    dictionary_t x( boost::move( dict));
    EXPECT_FALSE( x.empty());
}

TEST( Dictionary, Swap)
{
    dictionary_t dict;
    dict.insert()( dictionary_t::key_type( "key1"), dictionary_t::value_type( 7.0f))
                 ( "key2", dictionary_t::value_type( string8_t( "xxxyyy")))
                 ( "key3", dictionary_t::value_type( string8_t( "another_string")));

    dictionary_t x;
    x.swap( dict);

    EXPECT_TRUE( dict.empty());
    EXPECT_EQ( x.size(), 3);
}

TEST( Dictionary, Inserter)
{
    variant_t seven( 7.0f);

    dictionary_t dict;
    dict.insert()( dictionary_t::key_type( "key1"), seven)
                 ( "key2", variant_t( "xxxyyy"))
                 ( "key3", "another_string");

    EXPECT_EQ( dict.size(), 3);

    EXPECT_EQ( dict[dictionary_t::key_type( "key1")], seven);
    EXPECT_EQ( dict[dictionary_t::key_type( "key3")], variant_t( "another_string"));
}

TEST( Dictionary, GetValue)
{
    dictionary_t dict;
    dict.insert()( "key1", 11)
                 ( "key2", "xxxyyy")
                 ( "key3", point3f_t());

    // get + refs
    {

    }

    // get optional
    {

    }

    // get + pointers
    {
        const string8_t *x = get<string8_t>( &dict, dictionary_t::key_type( "key2"));
        EXPECT_TRUE( x);

        const int *y = get<boost::int32_t>( &dict, dictionary_t::key_type( "key7"));
        EXPECT_FALSE( y);
    }
}

TEST( Dictionary, Iterators)
{
    std::map<dictionary_t::key_type, variant_t> vals;
    vals[dictionary_t::key_type( "k0")] = variant_t( 7.0f);
    vals[dictionary_t::key_type( "k1")] = variant_t( 1.0f);
    vals[dictionary_t::key_type( "k2")] = variant_t( 4.0f);
    vals[dictionary_t::key_type( "k3")] = variant_t( 3.0f);

    dictionary_t dict;

    typedef std::pair<dictionary_t::key_type, variant_t> value_type;
    BOOST_FOREACH( const value_type& v, vals)
    {
        dict[v.first] = v.second;
    }

    typedef std::map<dictionary_t::key_type, variant_t>::const_iterator c_iter;
    c_iter comp_it( vals.begin());

    for( dictionary_t::const_iterator it( dict.begin()), e( dict.end()); it != e; ++it, ++comp_it)
    {
        //EXPECT_EQ( it->first, comp_it->first);
        //EXPECT_EQ( it->second, comp_it->second);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
