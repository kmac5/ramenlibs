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

#ifndef RAMEN_GL_FRAMEBUFFER_OBJ_HPP
#define	RAMEN_GL_FRAMEBUFFER_OBJ_HPP

#include<ramen/gl/renderbuffer_obj.hpp>
#include<ramen/gl/texture2d.hpp>

namespace ramen
{
namespace gl
{

class RAMEN_GL_API framebuffer_obj_t
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( framebuffer_obj_t)

public:

    framebuffer_obj_t();
    ~framebuffer_obj_t();

    // move constructor
    framebuffer_obj_t( BOOST_RV_REF( framebuffer_obj_t) other) : id_( 0)
    {
        swap( other);
    }

    // move assignment
    framebuffer_obj_t& operator=( BOOST_RV_REF( framebuffer_obj_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( framebuffer_obj_t& other)
    {
        boost::swap( id_, other.id_);
    }

    GLuint id() const { return id_;}

    void bind( GLenum target);
    void unbind( GLenum target);

    void attach_renderbuffer( const renderbuffer_obj_t& obj,
                              GLenum attachment_point);

    void detach_renderbuffer( GLenum attachment_point);

    void attach_texture( const texture2d_t& tx,
                         GLenum attachment_point,
                         int level = 0);

    void detach_texture( GLenum attachment_point);

    GLenum status() const;

private:

    GLuint id_;
};

template<class T>
inline void swap( framebuffer_obj_t& x, framebuffer_obj_t& y)
{
    x.swap( y);
}

} // gl
} // ramen

#endif
