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

#ifndef RAMEN_GL_TEXTURE2D_HPP
#define	RAMEN_GL_TEXTURE2D_HPP

#include<ramen/gl/gl.hpp>

#include<boost/swap.hpp>
#include<boost/move/move.hpp>

namespace ramen
{
namespace gl
{

class RAMEN_GL_API texture2d_t
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( texture2d_t)

public:

    texture2d_t();
    ~texture2d_t();

    // move constructor
    texture2d_t( BOOST_RV_REF( texture2d_t) other) : id_( 0)
    {
        swap( other);
    }

    // move assignment
    texture2d_t& operator=( BOOST_RV_REF( texture2d_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( texture2d_t& other)
    {
        boost::swap( id_, other.id_);
    }

    GLuint id() const { return id_;}

private:

    GLuint id_;
};

template<class T>
inline void swap( texture2d_t& x, texture2d_t& y)
{
    x.swap( y);
}

} // gl
} // ramen

#endif
