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

# Find Imath headers and libraries.
#
#  This module defines
# IMATH_INCLUDE_DIRS - where to find IMATH uncludes.
# IMATH_LIBRARIES    - List of libraries when using IMATH.
# IMATH_FOUND        - True if IMATH found.

# Look for the header file.
FIND_PATH( IMATH_INCLUDE_DIR NAMES OpenEXR/ImathVec.h)

# Look for the libraries.
FIND_LIBRARY( IMATH_IEX_LIBRARY NAMES Iex)
FIND_LIBRARY( IMATH_IEX_LIBRARY NAMES IexMath)
FIND_LIBRARY( IMATH_IEX_LIBRARY NAMES Iex)
FIND_LIBRARY( IMATH_MATH_LIBRARY NAMES Imath)

# handle the QUIETLY and REQUIRED arguments and set IMATH_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( IMATH DEFAULT_MSG	IMATH_IEX_LIBRARY
                                                      IMATH_MATH_LIBRARY
                                                      IMATH_INCLUDE_DIR
                                                      )
# Copy the results to the output variables.
IF(IMATH_FOUND)
    SET( IMATH_LIBRARIES ${IMATH_IEX_LIBRARY}
                         ${IMATH_MATH_LIBRARY}
                         )

    SET( IMATH_INCLUDE_DIRS ${IMATH_INCLUDE_DIR})
ELSE()
    SET( IMATH_LIBRARIES)
    SET( IMATH_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( IMATH_IEX_LIBRARY
                  IMATH_MATH_LIBRARY
                  IMATH_INCLUDE_DIR
                  )
