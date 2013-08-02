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

#ifndef RAMEN_GL_GL_HPP
#define	RAMEN_GL_GL_HPP

#include<ramen/gl/glew.hpp>

#ifdef __APPLE__
    #include<OpenGL/gl.h>
#else
    #if defined(WIN32) || defined(WIN64)
        #include<windows.h>
    #endif

    #include<GL/gl.h>
#endif

#include<ramen/core/exceptions.hpp>

namespace ramen
{
namespace gl
{

class RAMEN_GL_API exception : public core::exception
{
public:

    explicit exception( GLint err);

    virtual const char *what() const;

    GLint error() const { return error_;}

private:

    GLint error_;
};

RAMEN_GL_API void clear_errors();

inline void check_error()
{
    #ifndef NDEBUG
        GLint err = glGetError();

        if( err != GL_NO_ERROR)
            throw exception( err);
    #endif
}

inline void clear_color( GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha)
{
    glClearColor( red, green, blue, alpha);
    check_error();
}

inline void clear( GLbitfield mask)
{
    glClear( mask);
    check_error();
}

} // gl
} // ramen

#endif
