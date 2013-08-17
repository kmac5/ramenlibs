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

#include<ramen/gl/gl.hpp>

namespace ramen
{
namespace gl
{

gl_error::gl_error( GLint err) : exception(), error_( err)
{
    assert( error_ != GL_NO_ERROR);
}

const char *gl_error::what() const
{
    return reinterpret_cast<const char*>( gluErrorString( error_));
}

unsupported::unsupported( const char *extension) : exception()
{
    message_ = core::make_string( "extension ", extension, " not supported");
}

const char *unsupported::what() const
{
    return message_.c_str();
}

void clear_errors()
{
    while( glGetError() != GL_NO_ERROR)
        ;
}

} // gl
} // ramen
