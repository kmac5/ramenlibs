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

#ifndef RAMEN_GL_SHADER_HPP
#define	RAMEN_GL_SHADER_HPP

#include<ramen/gl/gl.hpp>

#include<istream>
#include<streambuf>

#include<boost/swap.hpp>
#include<boost/move/move.hpp>

#include<ramen/core/string8.hpp>

namespace ramen
{
namespace gl
{

class RAMEN_GL_API compile_error : public exception
{
public:

    compile_error();

    virtual const char *what() const;
};

class RAMEN_GL_API shader_t
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( shader_t)

public:

    shader_t( GLenum type, const char *src, std::size_t size);
    shader_t( GLenum type, const core::string8_t& src);

    shader_t( GLenum type, std::istream& is)
    {
        core::string8_t src;
        is.seekg( 0, std::ios::end);
        std::size_t size = is.tellg();
        src.reserve( size + 1);
        is.seekg( 0, std::ios::beg);
        src.assign(( std::istreambuf_iterator<char>( is)),
                     std::istreambuf_iterator<char>());
        init( type, src.c_str(), src.size());
    }

    ~shader_t();

    // move constructor
    shader_t( BOOST_RV_REF( shader_t) other) : id_( 0)
    {
        swap( other);
    }

    // move assignment
    shader_t& operator=( BOOST_RV_REF( shader_t) other)
    {
        swap( other);
        return *this;
    }

    void swap( shader_t& other)
    {
        boost::swap( id_, other.id_);
    }

    GLuint id() const { return id_;}

private:

    void init( GLenum type, const char *src, std::size_t size);

    GLuint id_;
};

template<class T>
inline void swap( shader_t& x, shader_t& y)
{
    x.swap( y);
}

} // gl
} // ramen

#endif
