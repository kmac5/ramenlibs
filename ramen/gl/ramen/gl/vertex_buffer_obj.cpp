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

#include<ramen/gl/vertex_buffer_obj.hpp>

#include<cassert>

namespace ramen
{
namespace gl
{

vertex_buffer_obj_t::vertex_buffer_obj_t() : id_( 0), bound_( false), mapped_( false)
{
    glGenBuffers( 1, &id_);
    check_error();
}

vertex_buffer_obj_t::~vertex_buffer_obj_t()
{
    assert( !bound_);
    assert( !mapped_);

    if( id_)
        glDeleteBuffers( 1, &id_);
}

void vertex_buffer_obj_t::bind( GLenum target)
{
    assert( id_);
    assert( !bound_);
    assert( !mapped_);

    glBindBuffer( target, id_);
    check_error();
    bound_ = true;
}

void vertex_buffer_obj_t::unbind()
{
    assert( bound_);
    assert( !mapped_);

    glBindBuffer( GL_ARRAY_BUFFER_ARB, 0);
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    bound_ = false;
}

void vertex_buffer_obj_t::reserve( GLenum target, GLsizei size, GLenum usage)
{
    assert( bound_);
    assert( !mapped_);

    glBufferData( target, size, 0, usage);
    check_error();
}

void vertex_buffer_obj_t::unmap_buffer( GLenum target)
{
    assert( bound_);
    assert( mapped_);

    glUnmapBuffer( target);
}

void vertex_buffer_obj_t::do_copy_from( GLenum target, GLsizei size, void *data, GLenum usage)
{
    assert( id_);
    assert( bound_);
    assert( !mapped_);

    glBufferData( target, size, data, usage);
    check_error();
}

void vertex_buffer_obj_t::do_update( GLenum target, GLint offset, GLsizei size, void *data)
{
    assert( id_);
    assert( bound_);
    assert( !mapped_);

    glBufferSubData( target, offset, size, data);
    check_error();
}

} // gl
} // ramen
