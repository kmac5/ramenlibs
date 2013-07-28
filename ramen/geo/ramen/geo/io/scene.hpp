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

#ifndef RAMEN_GEO_IO_SCENE_HPP
#define RAMEN_GEO_IO_SCENE_HPP

#include<ramen/geo/io/scene_fwd.hpp>

#include<ramen/geo/config.hpp>

#include<cassert>

#include<boost/move/move.hpp>

#include<ramen/geo/shape_vector_fwd.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

class RAMEN_GEO_IO_API scene_t
{
    BOOST_COPYABLE_AND_MOVABLE( scene_t)

public:

    scene_t();
    ~scene_t();

    // Copy constructor
    scene_t( const scene_t& other);

    // Copy assignment
    scene_t& operator=( BOOST_COPY_ASSIGN_REF( scene_t) other)
    {
        scene_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    scene_t( BOOST_RV_REF( scene_t) other)
    {
        assert( other.pimpl_);

        init();
        swap( other);
    }

    // Move assignment
    scene_t& operator=(BOOST_RV_REF( scene_t) other) //Move assignment
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    void swap( scene_t& other);

    const shape_vector_t& objects() const;
    shape_vector_t& objects();

private:

    void init();

    struct impl;
    impl *pimpl_;

    // TODO:
    // hierarchy...
    // materials...
    // sets...
    // ...
};

inline void swap( scene_t& x, scene_t& y)
{
    x.swap( y);
}

} // io
} // geo
} // ramen

#endif
