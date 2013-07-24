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

#include<gmock/gmock.h>

#include<boost/intrusive_ptr.hpp>

#include<ramen/core/ref_counted.hpp>

using namespace ramen::core;

using ::testing::InSequence;

class mock_ref_counted_t : public ref_counted_t
{
public:

    mock_ref_counted_t() : ref_counted_t() {}

    MOCK_METHOD0( destroyed, void());

    virtual ~mock_ref_counted_t()
    {
        destroyed();
    }

};

TEST( RefCounted, All)
{
    boost::intrusive_ptr<mock_ref_counted_t> obj( new mock_ref_counted_t());
    ASSERT_TRUE( obj->unique());
    {
        boost::intrusive_ptr<mock_ref_counted_t> obj2 = obj;
        ASSERT_FALSE( obj->unique());
    }
    ASSERT_TRUE( obj->unique());

    // mock
    {
        InSequence s;
        EXPECT_CALL( *obj, destroyed());
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
