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

# Find Antigrain headers and libraries.
#
#  AGG_INCLUDE_DIRS - where to find agg includes.
#  AGG_LIBRARIES    - List of libraries when using agg.
#  AGG_FOUND        - True if agg found.

FIND_PATH( AGG_INCLUDE_DIR NAMES agg/agg_bspline.h)
FIND_LIBRARY( AGG_LIBRARY NAMES agg)

INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( AGG DEFAULT_MSG AGG_LIBRARY AGG_INCLUDE_DIR)

# Copy the results to the output variables.
IF( AGG_FOUND)
    SET( AGG_LIBRARIES ${AGG_LIBRARY})
    SET( AGG_INCLUDE_DIRS ${AGG_INCLUDE_DIR})
ELSE()
    SET( AGG_LIBRARIES)
    SET( AGG_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( AGG_INCLUDE_DIR AGG_LIBRARY)
