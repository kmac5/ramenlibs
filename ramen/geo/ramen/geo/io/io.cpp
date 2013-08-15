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

#include<ramen/geo/io/io.hpp>

#include<vector>
#include<string>

#include<boost/foreach.hpp>

#include<ramen/geo/io/exceptions.hpp>

namespace ramen
{
namespace geo
{
namespace io
{
namespace
{

class formats_factory_t
{
public:

    static formats_factory_t& instance()
    {
        static formats_factory_t g_instance;
        return g_instance;
    }

    ~formats_factory_t()
    {
        BOOST_FOREACH( format_t *f, formats_)
            f->release();
    }

    void register_format( format_t *f)
    {
        int index = index_for_tag( f->tag());

        if( index != -1)
        {
            // a format with the same tag was already
            // registered, replace it.
            formats_[index]->release();
            formats_[index] = f;
        }
        else
        {
            // add the format to the end of the list.
            formats_.push_back( f);
        }
    }

    int index_for_tag( const core::name_t& tag)
    {
        for( int i = 0, e = formats_.size(); i < e; ++i)
        {
            if( formats_[i]->tag() == tag)
                return i;
        }

        return -1;
    }

    const format_t *format_for_extension( const char *filename, bool reading)
    {
        std::string ext( get_extension( filename));

        if( ext.empty())
            return 0;

        for( std::vector<format_t*>::const_reverse_iterator it( formats_.rbegin()), e( formats_.rend()); it != e; ++it)
        //BOOST_FOREACH( const format_t *f, formats_)
        {
            const format_t *f = *it;

            if( reading)
            {
                if( !f->can_read())
                    continue;
            }
            else
            {
                if( !f->can_write())
                    continue;
            }

            if( f->check_extension( ext.c_str()))
                return f;
        }

        return 0;
    }

private:

    // non-copyable
    formats_factory_t( const formats_factory_t&);
    formats_factory_t& operator=( const formats_factory_t&);

    std::string get_extension( const char *filename) const
    {
        std::string fname( filename);

        std::size_t n = fname.find_last_of( '.');

        if( n != std::string::npos)
            return std::string( fname, n + 1, std::string::npos);

        return std::string();
    }

    formats_factory_t() {}

    std::vector<format_t*> formats_;
};

} // unnamed

void register_io_format( core::auto_ptr_t<format_t>& format)
{
    assert( format.get());

    formats_factory_t::instance().register_format( format.release());
}

reader_t reader_for_file( const char *filename, const containers::dictionary_t& options)
{
    assert( filename);

    if( const format_t *f = formats_factory_t::instance().format_for_extension( filename, true))
    {
        core::auto_ptr_t<reader_interface_t> ri( f->create_reader( filename, options));
        if( ri)
            return reader_t( ri.release());
    }

    // TODO:
    // check file contents here.

    // fail if we can't open the file.
    throw io_exception( core::make_string( filename, ", unknown file format."));

    // keep dumb compilers happy.
    return reader_t( 0);
}

} // io
} // geo
} // ramen
