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

#include<ramen/gl/shader.hpp>

namespace ramen
{
namespace gl
{

compile_error::compile_error() : exception()
{
}

const char *compile_error::what() const
{
    return "compile error";
}

shader_t::shader_t( GLenum type, const char *src, std::size_t size)
{
    init( type, src, size);
}

shader_t::shader_t( GLenum type, const core::string8_t& src) : id_( 0)
{
    init( type, src.c_str(), src.size());
}

shader_t::~shader_t()
{
    if( id_)
        glDeleteShader( id_);
}

void shader_t::init( GLenum type, const char *src, std::size_t size)
{
    id_ = glCreateShader( type);
    check_error();

    GLint length = size;
    glShaderSource( id_, 1, reinterpret_cast<const GLchar**>( &src), &length);
    check_error();

    glCompileShader( id_);

    GLint compile_ok;
    glGetShaderiv( id_, GL_COMPILE_STATUS, &compile_ok);
    if( !compile_ok)
        throw compile_error();
}

} // gl
} // ramen
