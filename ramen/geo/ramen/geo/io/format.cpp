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

#include<ramen/geo/io/format.hpp>

#include<ramen/config/os.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

format_t::format_t() : flags_( 0) {}

format_t::~format_t() {}

reader_interface_t *format_t::create_reader( const char *filename,
                                             const core::dictionary_t &options) const
{
    return 0;
}

bool format_t::can_read() const
{
    return flags_ & can_read_bit;
}

bool format_t::can_write() const
{
    //return flags_ & can_write_bit;
    return false;
}

int format_t::compare( const char *str1, const char *str2,
                       int num_chars, bool case_sensitive) const
{
    if( case_sensitive)
        return strncmp( str1, str2, num_chars);
    else
    {
        #ifdef RAMEN_CONFIG_OS_WINDOWS
            return _strnicmp( str1, str2, num_chars);
        #else
            return strncasecmp( str1, str2, num_chars);
        #endif
    }
}

} // io
} // geo
} // ramen
