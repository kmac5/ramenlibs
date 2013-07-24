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

# Find pcre headers and libraries.
#
#  This module defines
#  PCRE_INCLUDE_DIRS - where to find PCRE uncludes.
#  PCRE_LIBRARIES    - List of libraries when using PCRE.
#  PCRE_FOUND        - True if PCRE found.

# Look for the header file.
FIND_PATH( PCRE_INCLUDE_DIR NAMES pcre.h)

# Look for the library.
FIND_LIBRARY( PCRE_LIBRARY NAMES pcre)
FIND_LIBRARY( PCRECPP_LIBRARY NAMES pcrecpp)
#FIND_LIBRARY( PCREPOSIX_LIBRARY NAMES pcreposix)

# handle the QUIETLY and REQUIRED arguments and set PCRE_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( PCRE DEFAULT_MSG PCRE_LIBRARY PCRECPP_LIBRARY PCRE_INCLUDE_DIR)

# Copy the results to the output variables.
IF( PCRE_FOUND)
    SET( PCRE_LIBRARIES ${PCRE_LIBRARY} ${PCRECPP_LIBRARY})
    SET( PCRE_INCLUDE_DIRS ${PCRE_INCLUDE_DIR})
ELSE()
    SET( PCRE_LIBRARIES)
    SET( PCRE_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( PCRE_INCLUDE_DIR PCRE_LIBRARY PCRECPP_LIBRARY)
