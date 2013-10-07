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

#ifndef RAMEN_GEO_IO_READER_HPP
#define RAMEN_GEO_IO_READER_HPP

#include<ramen/geo/config.hpp>

#include<boost/move/move.hpp>

#include<ramen/geo/io/reader_interface.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

class RAMEN_GEO_IO_API reader_t
{

    BOOST_MOVABLE_BUT_NOT_COPYABLE( reader_t)

public:

    explicit reader_t( reader_interface_t *model);
    ~reader_t();

    // Move constructor
    reader_t( BOOST_RV_REF( reader_t) other)
    {
        assert( other.model_);

        model_ = 0;
        swap( other);
    }

    // Move assignment
    reader_t& operator=( BOOST_RV_REF( reader_t) other)
    {
        assert( other.model_);

        swap( other);
        return *this;
    }

    void swap( reader_t& other);

    double start_time() const;
    double end_time() const;

    const shape_list_t& shape_list() const;

    const scene_t read() const;
    const scene_t read( const shape_list_filter_t& filter) const;

private:

    reader_interface_t *model_;
};

} // io
} // geo
} // ramen

#endif
