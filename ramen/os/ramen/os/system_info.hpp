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

#ifndef RAMEN_OS_SYSTEM_INFO_HPP
#define	RAMEN_OS_SYSTEM_INFO_HPP

#include<ramen/os/system_info_fwd.hpp>

#include<boost/cstdint.hpp>
#include<boost/filesystem/path.hpp>

#include<ramen/core/string8.hpp>

namespace ramen
{
namespace os
{

class RAMEN_OS_API system_info_t
{
public:

    system_info_t();
    ~system_info_t();

    boost::uint64_t ram_size() const
    {
        return ram_size_;
    }

    const core::string8_t& user_name() const
    {
        return user_name_;
    }

    const boost::filesystem::path& home_path() const
    {
        return home_path_;
    }

    const boost::filesystem::path& executable_path() const
    {
        return executable_path_;
    }

private:

    // non-copyable
    system_info_t( const system_info_t&);
    system_info_t& operator=( const system_info_t&);

    boost::uint64_t ram_size_;
    core::string8_t user_name_;
    boost::filesystem::path home_path_;
    boost::filesystem::path executable_path_;

    struct impl;
    impl *pimpl_;
};

} // os
} // ramen

#endif
