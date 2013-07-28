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

#include<ramen/geo/io/alembic/alembic_format.hpp>

#include<ramen/geo/io/io.hpp>

#include<ramen/geo/io/alembic/alembic_reader_model.hpp>

#include<Alembic/AbcGeom/All.h>
#include<Alembic/AbcCoreHDF5/All.h>

namespace ramen
{
namespace geo
{
namespace io
{

alembic_format_t::alembic_format_t() : tag_( "alembic")
{
    set_flags( can_read_bit);
}

void alembic_format_t::release() { delete this;}

const core::name_t& alembic_format_t::tag() const { return tag_;}

bool alembic_format_t::check_extension( const char *ext) const
{
    return compare( ext, "abc", 3, false) == 0;
}

reader_interface_t *alembic_format_t::create_reader( const char *filename,
                                                     const core::dictionary_t &options) const
{
    return new alembic_reader_model_t( filename, options);
}

// auto-register the format
namespace
{

bool register_alembic_format()
{
    core::auto_ptr_t<format_t> f( new alembic_format_t());
    register_io_format( f);
    return true;
}

static bool registered = register_alembic_format();

} // unnamed

} // io
} // geo
} // ramen
