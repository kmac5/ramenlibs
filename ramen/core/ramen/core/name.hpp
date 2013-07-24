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

#ifndef RAMEN_CORE_NAME_HPP
#define RAMEN_CORE_NAME_HPP

#include<ramen/core/name_fwd.hpp>

#include<iostream>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief Unique string class.
*/
class RAMEN_CORE_API name_t
{
public:

    name_t();
    explicit name_t( const char *str);

    name_t& operator=( const name_t& other);

    const char *c_str() const { return data_;}

    bool empty() const;

    bool operator==( const name_t& other) const { return data_ == other.data_;}
    bool operator!=( const name_t& other) const { return data_ != other.data_;}
    bool operator<( const name_t& other) const	{ return data_ < other.data_;}

    void swap( name_t& other)
    {
        const char *tmp = data_;
        data_ = other.data_;
        other.data_ = tmp;
    }

private:

    void init( const char *str);

    const char *data_;
};

inline void swap( name_t& x, name_t& y)
{
    x.swap( y);
}

inline std::ostream& operator<<( std::ostream& os, const name_t& name)
{
    return os << name.c_str();
}

} // core
} // ramen

#endif
