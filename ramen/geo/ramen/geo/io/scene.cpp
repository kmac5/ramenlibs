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

#include<ramen/geo/io/scene.hpp>

#include<ramen/core/memory.hpp>

#include<ramen/geo/shape_vector.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

struct scene_t::impl
{
    shape_vector_t objects_;
};

scene_t::scene_t()
{
    init();
    pimpl_ = new impl();
}

scene_t::~scene_t()
{
    delete pimpl_;
}

void scene_t::init()
{
    pimpl_ = 0;
}

scene_t::scene_t( const scene_t& other)
{
    pimpl_ = new impl( *other.pimpl_);
}

void scene_t::swap( scene_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

const shape_vector_t& scene_t::objects() const
{
    assert( pimpl_);

    return pimpl_->objects_;
}

shape_vector_t& scene_t::objects()
{
    assert( pimpl_);

    return pimpl_->objects_;
}

} // io
} // geo
} // ramen
