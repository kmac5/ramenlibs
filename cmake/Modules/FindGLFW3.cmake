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

#  Find GLFW3
#  Find GLFW3 headers and libraries.
#
#  This module defines
#  GLFW3_INCLUDE_DIRS - where to find GLFW3 includes.
#  GLFW3_LIBRARIES    - List of libraries when using GLFW3.
#  GLFW3_FOUND        - True if GLFW3 found.

# Look for the header file.
FIND_PATH( GLFW3_INCLUDE_DIR NAMES GL/glfw3.h)

# Look for the library.
FIND_LIBRARY( GLFW3_LIBRARY NAMES glfw)

# handle the QUIETLY and REQUIRED arguments and set GLFW3_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( GLFW3 DEFAULT_MSG GLFW3_LIBRARY GLFW3_INCLUDE_DIR)

# Copy the results to the output variables.
IF( GLFW3_FOUND)
    SET( GLFW3_LIBRARIES ${GLFW3_LIBRARY})
    SET( GLFW3_INCLUDE_DIRS ${GLFW3_INCLUDE_DIR})
ELSE()
    SET( GLFW3_LIBRARIES)
    SET( GLFW3_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( GLFW3_INCLUDE_DIR GLFW3_LIBRARY)
