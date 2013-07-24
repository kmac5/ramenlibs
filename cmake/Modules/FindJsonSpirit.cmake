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

#  Find json spirit
#  Find json spirit headers and libraries.
#
#  JSONSPIRIT_INCLUDE_DIRS - where to find json spirit includes.
#  JSONSPIRIT_LIBRARIES    - List of libraries when using json spirit.
#  JSONSPIRIT_FOUND         - True if json spirit found.

# Look for the header file.
FIND_PATH( JSONSPIRIT_INCLUDE_DIR NAMES ciere/json/value.hpp)

# Look for the libraries.
FIND_LIBRARY( JSONSPIRIT_LIBRARY NAMES json)

# handle the QUIETLY and REQUIRED arguments and set JSONSPIRIT_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( JsonSpirit DEFAULT_MSG  JSONSPIRIT_LIBRARY
                                                           JSONSPIRIT_INCLUDE_DIR
                                                           )
# Copy the results to the output variables.
IF( JSONSPIRIT_FOUND)
    SET( JSONSPIRIT_LIBRARIES ${JSONSPIRIT_LIBRARY})
    SET( JSONSPIRIT_INCLUDE_DIRS ${JSONSPIRIT_INCLUDE_DIR})
ELSE()
    SET( JSONSPIRIT_LIBRARIES)
    SET( JSONSPIRIT_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( JSONSPIRIT_LIBRARY JSONSPIRIT_INCLUDE_DIR)
