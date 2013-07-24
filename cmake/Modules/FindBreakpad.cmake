# Copyright (c) 2013 Esteban Tovagliari

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#  Find google's Breakpad headers and libraries.
#
#  This module defines
#  BREAKPAD_INCLUDE_DIRS
#  BREAKPAD_LIBRARY
#  BREAKPAD_CLIENT_LIBRARY
#  BREAKPAD_COMPILE_FLAGS
#  BREAKPAD_FOUND - True if BREAKPAD found.

# Look for the header file.
FIND_PATH( BREAKPAD_INCLUDE_DIR NAMES client/minidump_file_writer.h)

# Look for the library.
FIND_LIBRARY( BREAKPAD_LIB NAMES breakpad)
FIND_LIBRARY( BREAKPAD_CLIENT_LIB NAMES breakpad_client)

# handle the QUIETLY and REQUIRED arguments and set BREAKPAD_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( BREAKPAD DEFAULT_MSG BREAKPAD_LIB BREAKPAD_CLIENT_LIB BREAKPAD_INCLUDE_DIR)

# Copy the results to the output variables.
IF( BREAKPAD_FOUND)
    SET( BREAKPAD_LIBRARY ${BREAKPAD_LIB})
    SET( BREAKPAD_CLIENT_LIBRARY ${BREAKPAD_CLIENT_LIB})
    SET( BREAKPAD_INCLUDE_DIRS ${BREAKPAD_INCLUDE_DIR})
ELSE()
    SET( BREAKPAD_LIBRARY)
    SET( BREAKPAD_CLIENT_LIBRARY)
    SET( BREAKPAD_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( BREAKPAD_INCLUDE_DIR BREAKPAD_LIB BREAKPAD_CLIENT_LIB)
