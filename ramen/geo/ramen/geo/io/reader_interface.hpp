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

#ifndef RAMEN_GEO_IO_READER_INTERFACE_HPP
#define RAMEN_GEO_IO_READER_INTERFACE_HPP

#include<ramen/geo/config.hpp>

#include<ramen/core/memory.hpp>

#include<ramen/containers/dictionary.hpp>

#include<ramen/geo/io/exceptions.hpp>
#include<ramen/geo/io/scene.hpp>
#include<ramen/geo/io/shape_list.hpp>
#include<ramen/geo/global_names.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

/*
    TODO: as reader_interface_t holds data and possibly some
    code too, it's not an interface anymore. Find a new name.
*/

class RAMEN_GEO_IO_API reader_interface_t
{
public:

    virtual void release() = 0;

    const core::string8_t& filename() const       { return filename_;}
    const containers::dictionary_t& options() const     { return options_;}

    float start_time() const;
    float end_time() const;

    const shape_list_t& shape_list() const;

    virtual const scene_t read( const shape_list_filter_t& filter) = 0;

protected:

    reader_interface_t( const char *filename, const containers::dictionary_t& options);
    virtual ~reader_interface_t();

    core::string8_t filename_;
    const containers::dictionary_t& options_;

    float start_time_;
    float end_time_;
    shape_list_t shape_list_;
};

} // io
} // geo

namespace core
{

template<>
struct pointer_traits<geo::io::reader_interface_t*>
{
    typedef geo::io::reader_interface_t     element_type;
    typedef geo::io::reader_interface_t*    pointer_type;
    typedef const pointer_type              const_pointer_type;

    static bool empty_ptr( geo::io::reader_interface_t *x)  { return x == 0;}
    static void delete_ptr( geo::io::reader_interface_t *x) { x->release();}
};

} // core
} // ramen

#endif
