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

#ifndef RAMEN_OS_IMPL_SYSTEM_INFO_WIN_HPP
#define RAMEN_OS_IMPL_SYSTEM_INFO_WIN_HPP

#error "Windows not supported yet"

namespace ramen
{
namespace os
{

struct system_info_t::impl
{
    explicit impl( system_info_t& self)
    {
        /*
            BOOL WINAPI GetUserName( _Out_    LPTSTR lpBuffer,
                                     _Inout_  LPDWORD lpnSize);
        */
        /*
            // executable path
            DWORD WINAPI GetModuleFileName( _In_opt_  HMODULE hModule,
                                            _Out_     LPTSTR lpFilename,
                                            _In_      DWORD nSize);
        */
        /*
            #include "Userenv.h"
            #pragma comment(lib, "userenv.lib")
            CString GetUserHomeDir()
            {
               TCHAR szHomeDirBuf[MAX_PATH] = { 0 };

               HANDLE hToken = 0;
               VERIFY( OpenProcessToken( GetCurrentProcess(), TOKEN_QUERY, &hToken ));

               DWORD BufSize = MAX_PATH;
               VERIFY( GetUserProfileDirectory( hToken, szHomeDirBuf, &BufSize ));

               CloseHandle( hToken );
               return CString( szHomeDirBuf );
            }
        */
        /*
            // RAM Size

            //  Sample output:
            //  There is       51 percent of memory in use.
            //  There are 2029968 total KB of physical memory.
            //  There are  987388 free  KB of physical memory.
            //  There are 3884620 total KB of paging file.
            //  There are 2799776 free  KB of paging file.
            //  There are 2097024 total KB of virtual memory.
            //  There are 2084876 free  KB of virtual memory.
            //  There are       0 free  KB of extended memory.

            #include <windows.h>
            #include <stdio.h>
            #include <tchar.h>

            // Use to convert bytes to KB
            #define DIV 1024

            // Specify the width of the field in which to print the numbers.
            // The asterisk in the format specifier "%*I64d" takes an integer
            // argument and uses it to pad and right justify the number.
            #define WIDTH 7

            void _tmain()
            {
              MEMORYSTATUSEX statex;

              statex.dwLength = sizeof (statex);

              GlobalMemoryStatusEx (&statex);

              _tprintf (TEXT("There is  %*ld percent of memory in use.\n"),
                        WIDTH, statex.dwMemoryLoad);
              _tprintf (TEXT("There are %*I64d total KB of physical memory.\n"),
                        WIDTH, statex.ullTotalPhys/DIV);
              _tprintf (TEXT("There are %*I64d free  KB of physical memory.\n"),
                        WIDTH, statex.ullAvailPhys/DIV);
              _tprintf (TEXT("There are %*I64d total KB of paging file.\n"),
                        WIDTH, statex.ullTotalPageFile/DIV);
              _tprintf (TEXT("There are %*I64d free  KB of paging file.\n"),
                        WIDTH, statex.ullAvailPageFile/DIV);
              _tprintf (TEXT("There are %*I64d total KB of virtual memory.\n"),
                        WIDTH, statex.ullTotalVirtual/DIV);
              _tprintf (TEXT("There are %*I64d free  KB of virtual memory.\n"),
                        WIDTH, statex.ullAvailVirtual/DIV);

              // Show the amount of extended memory available.

              _tprintf (TEXT("There are %*I64d free  KB of extended memory.\n"),
                        WIDTH, statex.ullAvailExtendedVirtual/DIV);
            }
        */
    }
};

} // os
} // ramen

#endif
