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

#ifndef RAMEN_GEO_ALEMBIC_FORMAT_HPP
#define RAMEN_GEO_ALEMBIC_FORMAT_HPP

#include<ramen/geo/io/format.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

class alembic_format_t : public format_t
{
public:

    alembic_format_t();

    virtual void release();

    virtual const core::name_t& tag() const;

    virtual bool check_extension( const char *ext) const;

    virtual reader_interface_t *create_reader( const char *filename,
                                               const containers::dictionary_t& options) const;

private:

    core::name_t tag_;
};

} // io
} // geo
} // ramen

#endif
