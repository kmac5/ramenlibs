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

#ifndef RAMEN_OS_IMPL_SYSTEM_INFO_OSX_HPP
#define RAMEN_OS_IMPL_SYSTEM_INFO_OSX_HPP

#include<pwd.h>
#include<sys/sysctl.h>

namespace ramen
{
namespace os
{

struct system_info_t::impl
{
    explicit impl( system_info_t& self)
    {
        // user name & home path
        struct passwd *p = getpwuid( geteuid());
        self.user_name_ = p->pw_name;
        self.home_path_ = boost::filesystem::path( p->pw_dir);

        // ram size
        {
            int mib[2];
            mib[0] = CTL_HW;
            mib[1] = HW_MEMSIZE;
            int64_t size = 0;
            size_t len = sizeof( size);

            if( sysctl( mib, 2, &size, &len, NULL, 0) == 0)
                self.ram_size_ = size;
        }
    }
};

} // os
} // ramen

#endif
