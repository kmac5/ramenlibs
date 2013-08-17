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

#include<ramen/gl/framebuffer_obj.hpp>

namespace ramen
{
namespace gl
{

framebuffer_obj_t::framebuffer_obj_t() : id_( 0)
{
    if( !GLEW_EXT_framebuffer_object)
        throw unsupported( "EXT_framebuffer_object");

    glGenFramebuffersEXT( 1, &id_);
    check_error();
}

framebuffer_obj_t::~framebuffer_obj_t()
{
    if( id_)
        glDeleteFramebuffersEXT( 1, &id_);
}

void framebuffer_obj_t::bind( GLenum target)
{
    glBindFramebufferEXT( target, id_);
    check_error();
}

void framebuffer_obj_t::unbind( GLenum target)
{
    glBindFramebufferEXT( target, 0);
}

void framebuffer_obj_t::attach_renderbuffer( const renderbuffer_obj_t& obj,
                                             GLenum attachment_point)
{
    glFramebufferRenderbufferEXT( GL_FRAMEBUFFER,
                                  attachment_point,
                                  GL_RENDERBUFFER,
                                  obj.id());
    check_error();
}

void framebuffer_obj_t::detach_renderbuffer(GLenum attachment_point)
{
    glFramebufferRenderbufferEXT( GL_FRAMEBUFFER,
                                  attachment_point,
                                  GL_RENDERBUFFER,
                                  0);
    check_error();
}

void framebuffer_obj_t::attach_texture( const texture2d_t& tx,
                                        GLenum attachment_point,
                                        int level)
{
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER,
                               attachment_point,
                               GL_TEXTURE_2D,
                               tx.id(),
                               level);
    check_error();
}

void framebuffer_obj_t::detach_texture( GLenum attachment_point)
{
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER,
                               attachment_point,
                               GL_TEXTURE_2D,
                               0, 0);
    check_error();
}

GLenum framebuffer_obj_t::status() const
{
    return glCheckFramebufferStatusEXT( GL_FRAMEBUFFER);
}

} // gl
} // ramen
