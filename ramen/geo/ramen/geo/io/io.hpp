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

#ifndef RAMEN_GEO_IO_IO_HPP
#define RAMEN_GEO_IO_IO_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/io/format.hpp>
#include<ramen/geo/io/reader.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

RAMEN_GEO_IO_API void register_io_format( core::auto_ptr_t<format_t>& format);

/*
 *  Input options:
 *  --------------
 *
 *      Name                Type        Default             Desc
 *      ----                ----        -------             ----
 *
 *      time                float       file start time     for animated formats, the time at which to read the geo.
 *      paths_as_names      bool        false               for hierarchical formats, use the full path to the obj as the name.
 *      apply_transforms    bool        true                for hierarchical formats, apply the transforms to the geo.
 *      read_arbitrary      bool        true                read arbitrary attributes.
 *
 */

RAMEN_GEO_IO_API reader_t reader_for_file( const char *filename,
                                          const core::dictionary_t& options);

} // io
} // geo
} // ramen

#endif
