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

#   Find OpenSubdiv headers and libraries.
#
#  This module defines
#  OpenSubdiv_INCLUDE_DIRS - where to find OpenSubdiv includes.
#  OpenSubdiv_LIBRARIES    - List of libraries when using OpenSubdiv.
#  OpenSubdiv_FOUND        - True if OpenSubdiv found.

# Look for the header file.
FIND_PATH( OPENSUBDIV_INCLUDE_DIR NAMES osd/vertex.h)

# Look for the library.
FIND_LIBRARY( OPENSUBDIV_LIBRARY NAMES osd)
FIND_LIBRARY( OPENSUBDIV_CPU_LIBRARY NAMES osdCPU)
FIND_LIBRARY( OPENSUBDIV_GPU_LIBRARY NAMES osdGPU)

# handle the QUIETLY and REQUIRED arguments and set OPENSUBDIV_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OPENSUBDIV DEFAULT_MSG
                                    OPENSUBDIV_LIBRARY
                                    OPENSUBDIV_CPU_LIBRARY
                                    OPENSUBDIV_GPU_LIBRARY
                                    OPENSUBDIV_INCLUDE_DIR)

# compilation flags
SET( OPENSUBDIV_COMPILE_FLAGS)

# Copy the results to the output variables.
IF( OPENSUBDIV_FOUND)
    SET( OPENSUBDIV_LIBRARIES ${OPENSUBDIV_LIBRARY}
                              ${OPENSUBDIV_CPU_LIBRARY}
                              ${OPENSUBDIV_GPU_LIBRARY}
                              )

    SET( OPENSUBDIV_INCLUDE_DIRS ${OPENSUBDIV_INCLUDE_DIR})
ELSE()
    SET( OPENSUBDIV_LIBRARIES)
    SET( OPENSUBDIV_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( OPENSUBDIV_INCLUDE_DIR OPENSUBDIV_LIBRARY OPENSUBDIV_CPU_LIBRARY OPENSUBDIV_GPU_LIBRARY)
