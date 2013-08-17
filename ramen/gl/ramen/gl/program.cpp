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

#include<ramen/gl/program.hpp>

namespace ramen
{
namespace gl
{

link_error::link_error() : exception()
{
}

const char *link_error::what() const
{
    return "link error";
}

program_t::program_t() : id_( 0)
{
    id_ = glCreateProgram();
    check_error();
}

program_t::program_t( const shader_t& vertex, const shader_t& fragment) : id_( 0)
{
    id_ = glCreateProgram();
    check_error();

    attach( vertex);
    attach( fragment);
}

program_t::~program_t()
{
    if( id_)
        glDeleteProgram( id_);
}

void program_t::attach( const shader_t& shader)
{
    glAttachShader( id_, shader.id());
    check_error();
}

void program_t::detach( const shader_t& shader)
{
    glDetachShader( id_, shader.id());
    check_error();
}

void program_t::link()
{
    glLinkProgram( id_);
    check_error();

    GLint link_ok;
    glGetProgramiv( id_, GL_LINK_STATUS, &link_ok);
    if( !link_ok)
        throw link_error();
}

void program_t::use()
{
    glUseProgram( id_);
    check_error();
}

void program_t::unuse()
{
    glUseProgram( 0);
}

} // gl
} // ramen
