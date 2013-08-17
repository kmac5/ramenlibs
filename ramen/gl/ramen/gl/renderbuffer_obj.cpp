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

#include<ramen/gl/renderbuffer_obj.hpp>

namespace ramen
{
namespace gl
{

renderbuffer_obj_t::renderbuffer_obj_t() : id_( 0)
{
    if( !GLEW_EXT_framebuffer_object)
        throw unsupported( "EXT_framebuffer_object");

    glGenRenderbuffersEXT( 1, &id_);
    check_error();
}

renderbuffer_obj_t::~renderbuffer_obj_t()
{
    if( id_)
        glDeleteRenderbuffersEXT( 1, &id_);
}

void renderbuffer_obj_t::bind( GLenum target)
{
    glBindRenderbufferEXT( target, id_);
    check_error();
}

void renderbuffer_obj_t::unbind( GLenum target)
{
    glBindRenderbufferEXT( target, 0);
}

void renderbuffer_obj_t::alloc_storage( GLenum format, GLsizei width, GLsizei height)
{
    glRenderbufferStorageEXT( GL_RENDERBUFFER, format, width, height);
    check_error();
}

} // gl
} // ramen
