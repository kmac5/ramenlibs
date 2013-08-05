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

#include<ramen/core/name.hpp>

#include<cassert>
#include<string>
#include<algorithm>

#include<boost/unordered_set.hpp>

//#include<boost/thread/mutex.hpp>
//#include<boost/thread/locks.hpp>

#include"tinythread.h"

#ifdef RAMEN_USE_INTERPROCESS_NAMES
    #include<boost/interprocess/detail/intermodule_singleton.hpp>
#endif

namespace ramen
{
namespace core
{
namespace
{

struct name_pool_t
{
    typedef boost::unordered::unordered_set<std::string> name_set_type;
    typedef name_set_type::const_iterator	const_iterator;
    typedef name_set_type::iterator			iterator;

    //typedef boost::mutex                    mutex_type;
    //typedef boost::lock_guard<mutex_type>   lock_type;

    typedef tthread::mutex                  mutex_type;
    typedef tthread::lock_guard<mutex_type> lock_type;

    name_pool_t()
    {
        g_empty_string = add( std::string( ""));
    }

    const char *add( const std::string& str)
    {
        lock_type lock( mutex_);

        const_iterator it( names_.find( str));

        if( it != names_.end())
            return it->c_str();

        std::pair<iterator, bool> result = names_.insert( str);
        return result.first->c_str();
    }

    mutex_type mutex_;
    name_set_type names_;
    const char *g_empty_string;
};

#ifdef RAMEN_USE_INTERPROCESS_NAMES
    typedef boost::interprocess::ipcdetail::intermodule_singleton<name_pool_t> singleton_name_pool_holder_t;
#else
    struct singleton_name_pool_holder_t
    {
        static name_pool_t& get()
        {
            static name_pool_t pool;
            return pool;
        }
    };
#endif

} // unnamed

name_t::name_t() { init( "");}

name_t::name_t( const char *str)
{
    assert( str);

    init( str);
}

void name_t::init( const char *str)
{
    data_ = singleton_name_pool_holder_t::get().add( std::string( str));
}

name_t& name_t::operator=( const name_t& other)
{
    using std::swap;

    name_t tmp( other);
    swap( *this, tmp);
    return *this;
}

bool name_t::empty() const
{
    return c_str() == singleton_name_pool_holder_t::get().g_empty_string;
}

} // core
} // ramen
