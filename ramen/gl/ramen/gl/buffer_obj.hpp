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

#ifndef RAMEN_GL_BUFFER_OBJ_HPP
#define	RAMEN_GL_BUFFER_OBJ_HPP

#include<ramen/gl/gl.hpp>

#include<boost/swap.hpp>
#include<boost/move/move.hpp>

namespace ramen
{
namespace gl
{

class RAMEN_GL_API buffer_obj_t
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( buffer_obj_t)

public:

    buffer_obj_t();
    ~buffer_obj_t();

    // move constructor
    buffer_obj_t( BOOST_RV_REF( buffer_obj_t) other) : id_( 0)
    {
        swap( other);
    }

    // move assignment
    buffer_obj_t& operator=( BOOST_RV_REF( buffer_obj_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( buffer_obj_t& other)
    {
        boost::swap( id_, other.id_);
    }

    GLuint id() const { return id_;}

    void bind( GLenum target);
    void unbind();

    void reserve( GLenum target, GLsizei size, GLenum usage);

    template<class T>
    void copy_from( GLenum target, GLsizei size, T *data, GLenum usage)
    {
        do_copy_from( target, size * sizeof( T), reinterpret_cast<void*>( data), usage);
    }

    template<class T>
    void update( GLenum target, GLint offset, GLsizei size, T *data)
    {
        do_update( target, offset * sizeof( T), size * sizeof( T), reinterpret_cast<void*>( data));
    }

    template<class T>
    T *map_buffer( GLenum target, GLenum access)
    {
        void *ptr = glMapBuffer( target, access);
        check_error();
        return reinterpret_cast<T*>( ptr);
    }

    void unmap_buffer( GLenum target);

private:

    void do_copy_from( GLenum target, GLsizei size, void *data, GLenum usage);
    void do_update( GLenum target, GLint offset, GLsizei size, void *data);

    GLuint id_;
};

template<class T>
inline void swap( buffer_obj_t& x, buffer_obj_t& y)
{
    x.swap( y);
}

} // gl
} // ramen

#endif
