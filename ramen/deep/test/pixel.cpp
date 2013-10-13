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

#include<ramen/deep/pixel.hpp>

#include<boost/algorithm/cxx11/is_sorted.hpp>

using namespace ramen::core;
using namespace ramen::color;
using namespace ramen::math;
using namespace ramen::arrays;
using namespace ramen::deep;

TEST( DeepPixel, Construct)
{
    pixel_t p;
}

TEST( DeepPixel, Sort)
{
    pixel_t p;
    p.push_back_discrete_sample( 1.0f, color3f_t( 0.2f));    
    p.push_back_discrete_sample( 4.0f, color3f_t( 0.7f));    
    p.push_back_discrete_sample( 7.0f, color3f_t( 0.3f));    
    p.push_back_discrete_sample( 2.0f, color3f_t( 0.4f));    
    p.push_back_discrete_sample( 1.0f, color3f_t( 0.1f));
    p.sort();
    
    EXPECT_TRUE( boost::algorithm::is_sorted( const_array_ref_t<float>( p.array( name_t( "Z")))));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
