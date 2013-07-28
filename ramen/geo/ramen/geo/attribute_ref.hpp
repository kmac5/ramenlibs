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

#ifndef RAMEN_GEO_ATTRIBUTE_REF_HPP
#define RAMEN_GEO_ATTRIBUTE_REF_HPP

#include<ramen/geo/config.hpp>

#include<ramen/arrays/array_ref.hpp>

namespace ramen
{
namespace geo
{

/*!
\brief Constant attribute reference.
*/
struct RAMEN_GEO_API const_attribute_ref_t
{
    // for safe bool
    operator int() const;

public:

    const_attribute_ref_t( const arrays::array_t *data);

    bool valid() const;

    const arrays::array_t& array() const;

    core::type_t array_type() const;

    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return ( data_ != 0);}
    bool operator!() const throw();

protected:

    arrays::array_t *data_;
};

/*!
\brief Non-constant attribute reference.
*/
struct RAMEN_GEO_API attribute_ref_t : public const_attribute_ref_t
{
    // for safe bool
    operator int() const;

public:

    attribute_ref_t( arrays::array_t *data);

    arrays::array_t& array();

    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return ( data_ != 0);}
    bool operator!() const throw();
};

} // geo
} // ramen

#endif
