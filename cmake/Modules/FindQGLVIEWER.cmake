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

# Find QGLVIEWER headers and libraries.
#
#  QGLVIEWER_INCLUDE_DIRS - where to find QGLVIEWER includes.
#  QGLVIEWER_LIBRARIES    - List of libraries when using QGLVIEWER.
#  QGLVIEWER_FOUND        - True if QGLVIEWER found.

FIND_PATH( QGLVIEWER_INCLUDE_DIR NAMES QGLViewer/qglviewer.h)
FIND_LIBRARY( QGLVIEWER_LIBRARY NAMES QGLViewer)

INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( QGLVIEWER DEFAULT_MSG QGLVIEWER_LIBRARY QGLVIEWER_INCLUDE_DIR)

# Copy the results to the output variables.
IF( QGLVIEWER_FOUND)
    SET( QGLVIEWER_LIBRARIES ${QGLVIEWER_LIBRARY})
    SET( QGLVIEWER_INCLUDE_DIRS ${QGLVIEWER_INCLUDE_DIR})
ELSE()
    SET( QGLVIEWER_LIBRARIES)
    SET( QGLVIEWER_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( QGLVIEWER_INCLUDE_DIR QGLVIEWER_LIBRARY)
