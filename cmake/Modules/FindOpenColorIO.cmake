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

# Find OpenColorIO headers and libraries.
#
#  This module defines
#  OPENCOLORIO_INCLUDE_DIRS - where to find OpenColorIO uncludes.
#  OPENCOLORIO_LIBRARIES    - List of libraries when using OpenColorIO.
#  OPENCOLORIO_FOUND        - True if OpenColorIO found.

# Look for the header file.
FIND_PATH( OPENCOLORIO_INCLUDE_DIR NAMES OpenColorIO/OpenColorIO.h)

# Look for the library.
FIND_LIBRARY( OPENCOLORIO_LIBRARY NAMES OpenColorIO)

# handle the QUIETLY and REQUIRED arguments and set OPENCOLORIO_FOUND to TRUE if all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OPENCOLORIO DEFAULT_MSG OPENCOLORIO_LIBRARY OPENCOLORIO_INCLUDE_DIR)

# Copy the results to the output variables.
IF( OPENCOLORIO_FOUND)
    SET( OPENCOLORIO_LIBRARIES ${OPENCOLORIO_LIBRARY})
    SET( OPENCOLORIO_INCLUDE_DIRS ${OPENCOLORIO_INCLUDE_DIR})
ELSE()
    SET( OPENCOLORIO_LIBRARIES)
    SET( OPENCOLORIO_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( OPENCOLORIO_INCLUDE_DIR OPENCOLORIO_LIBRARY)
