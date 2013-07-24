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

#  Find OpenNurbs headers and libraries.
#
#  This module defines
#  OPENNURBS_INCLUDE_DIRS - where to find OpenNurbs uncludes.
#  OPENNURBS_LIBRARIES    - List of libraries when using OpenNurbs.
#  OPENNURBS_FOUND        - True if OpenNurbs found.

# Look for the header file.
FIND_PATH( OPENNURBS_INCLUDE_DIR NAMES opennurbs.h)

# Look for the library.
FIND_LIBRARY( OPENNURBS_LIBRARY NAMES openNURBS)

# handle the QUIETLY and REQUIRED arguments and set OPENNURBS_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OPENNURBS DEFAULT_MSG OPENNURBS_LIBRARY OPENNURBS_INCLUDE_DIR)

# Copy the results to the output variables.
IF( OPENNURBS_FOUND)
    SET( OPENNURBS_LIBRARIES ${OPENNURBS_LIBRARY})
    SET( OPENNURBS_INCLUDE_DIRS ${OPENNURBS_INCLUDE_DIR})
ELSE()
    SET( OPENNURBS_LIBRARIES)
    SET( OPENNURBS_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( OPENNURBS_INCLUDE_DIR OPENNURBS_LIBRARY)
