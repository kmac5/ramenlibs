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

#include<ramen/geo/io/reader.hpp>

#include<algorithm>

namespace ramen
{
namespace geo
{
namespace io
{

reader_t::reader_t( reader_interface_t *model) : model_( model)
{
    assert( model_);
}

reader_t::~reader_t()
{
    if( model_)
        model_->release();
}

void reader_t::swap( reader_t& other)
{
    std::swap( model_, other.model_);
}

float reader_t::start_time() const
{
    assert( model_);

    return model_->start_time();
}

float reader_t::end_time() const
{
    assert( model_);

    return model_->end_time();
}

const shape_list_t& reader_t::shape_list() const
{
    assert( model_);

    return model_->shape_list();
}

const scene_t reader_t::read() const
{
    shape_list_filter_t filter;
    return read( filter);
}

const scene_t reader_t::read( const shape_list_filter_t& filter) const
{
    assert( model_);

    return model_->read( filter);
}

} // io
} // geo
} // ramen
