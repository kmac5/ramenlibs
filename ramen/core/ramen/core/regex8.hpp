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

#ifndef RAMEN_CORE_REGEX8_HPP
#define RAMEN_CORE_REGEX8_HPP

#include<ramen/core/final.hpp>

#include<ramen/core/string8.hpp>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief UTF8 regular expression class.
*/
class RAMEN_CORE_API regex8_t : RAMEN_CORE_FINAL( regex8_t)
{
    BOOST_COPYABLE_AND_MOVABLE( regex8_t)

public:

    explicit regex8_t( const string8_t& pattern);

    ~regex8_t();

    // Copy constructor
    regex8_t( const regex8_t& other);

    // Copy assignment
    regex8_t& operator=( BOOST_COPY_ASSIGN_REF( regex8_t) other)
    {
        assert( other.pimpl_);

        regex8_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    regex8_t( BOOST_RV_REF( regex8_t) other) : pimpl_( 0)
    {
        assert( other.pimpl_);

        swap( other);
    }

    // Move assignment
    regex8_t& operator=( BOOST_RV_REF( regex8_t) other)
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    void swap( regex8_t& other);

    bool full_match( const string8_t& str) const;
    bool full_match( const char *str) const;

private:

    struct impl;
    impl *pimpl_;
};

inline void swap( regex8_t& x, regex8_t& y)
{
    x.swap( y);
}

} // core
} // ramen

#endif
