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

#include<ramen/geo/attribute_table.hpp>
using namespace ramen;
using namespace ramen::geo;

/*
BOOST_AUTO_TEST_CASE( attribute_table_test)
{
    attribute_table_t attr;

    name_t P( "P");
    attr.insert( P, float_k);
    {
        BOOST_CHECK( attr.has_attribute( P));
        BOOST_CHECK( attr.attribute_type( P) == float_k);

        const_attribute_ref_t r_ref( attr.get_const_attribute_ref( P));
        BOOST_CHECK( r_ref.valid());
        BOOST_CHECK( r_ref.array().empty());
    }

    array_t values( float_k);
    array_ref_t<float> ref( &values);

    for( int i = 0; i < 30; ++i)
        ref.push_back( 77.0f);

    name_t Q( "Q");
    attr.insert( Q, values);
    {
        BOOST_CHECK( !array_was_moved( values));
        BOOST_CHECK( attr.has_attribute( Q));
        BOOST_CHECK( attr.attribute_type( Q) == float_k);

        const_attribute_ref_t r_ref( attr.get_const_attribute_ref( Q));
        BOOST_CHECK( r_ref.valid());
        BOOST_CHECK( r_ref.array() == values);
    }

    name_t R( "R");
    attr.insert( R, float_k);
    {
        BOOST_CHECK( attr.has_attribute( R));
        BOOST_CHECK( attr.attribute_type( R) == float_k);

        const_attribute_ref_t r_ref( attr.get_const_attribute_ref( R));
        BOOST_CHECK( r_ref.valid());
        BOOST_CHECK( r_ref.array().empty());
    }
    {
        attribute_ref_t r_ref( attr.get_attribute_ref( R));
        BOOST_CHECK( r_ref.valid());

        array_ref_t<float> z_ref( &( r_ref.array()));
        z_ref.push_back( 11.0f);
        z_ref.push_back( 10.0f);
        z_ref.push_back( 12.0f);
        z_ref.push_back( 14.0f);
        int size = z_ref.size();

        BOOST_CHECK( attr.get_const_attribute_ref( R).array().size() == size);
    }

    name_t T( "T");
    attr.insert( T, boost::move( values));
    {
        BOOST_CHECK( array_was_moved( values));
        BOOST_CHECK( attr.has_attribute( T));
        BOOST_CHECK( attr.attribute_type( T) == float_k);

        const_attribute_ref_t r_ref( attr.get_const_attribute_ref( Q));
        BOOST_CHECK( r_ref.valid());
    }
}
*/

TEST( AttributeTable, All)
{
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
