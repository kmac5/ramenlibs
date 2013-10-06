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

#ifndef RAMEN_GEO_IO_FORMAT_HPP
#define RAMEN_GEO_IO_FORMAT_HPP

#include<ramen/geo/config.hpp>

#include<cstring>

#include<ramen/core/name.hpp>

#include<ramen/containers/string8_vector_fwd.hpp>

#include<ramen/geo/io/reader_interface.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

class RAMEN_GEO_IO_API format_t
{
public:

    virtual void release() = 0;

    virtual const core::name_t& tag() const = 0;

    bool can_read() const;
    bool can_write() const;

    virtual void add_extensions( containers::string8_vector_t& ext_list) const = 0;
    virtual bool check_extension( const char *ext) const = 0;

    virtual reader_interface_t *create_reader( const char *filename,
                                               const containers::dictionary_t& options) const;
protected:

    format_t();
    virtual ~format_t();

    int compare( const char *str1, const char *str2,
                 int num_chars, bool case_sensitive) const;

    // flags
    enum
    {
        can_read_bit  = 1,
        can_write_bit = 1 << 1
    };

    void set_flags( boost::uint32_t f) { flags_ = f;}

private:

    boost::uint32_t flags_;
};

} // io
} // geo

namespace core
{

template<>
struct pointer_traits<geo::io::format_t*>
{
    typedef geo::io::format_t     element_type;
    typedef geo::io::format_t*    pointer_type;
    typedef const pointer_type    const_pointer_type;

    static bool empty_ptr( geo::io::format_t *x)  { return x == 0;}
    static void delete_ptr( geo::io::format_t *x) { x->release();}
};

} // core
} // ramen

#endif
