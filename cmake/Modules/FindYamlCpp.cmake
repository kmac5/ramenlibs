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

#  Find YAMLCPP headers and libraries.
#
#  This module defines
#  YAMLCPP_INCLUDE_DIRS - where to find yaml-cpp includes.
#  YAMLCPP_LIBRARIES    - List of libraries when using yaml-cpp.
#  YAMLCPP_FOUND        - True if yaml-cpp found.

# Look for the header file.
FIND_PATH( YAMLCPP_INCLUDE_DIR NAMES yaml-cpp/yaml.h)

# Look for the library.
FIND_LIBRARY( YAMLCPP_LIBRARY NAMES yaml-cpp)

# handle the QUIETLY and REQUIRED arguments and set YAMLCPP_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( YAMLCPP DEFAULT_MSG YAMLCPP_LIBRARY YAMLCPP_INCLUDE_DIR)

# our compilation flags
SET( YAMLCPP_COMPILE_FLAGS)

# Copy the results to the output variables.
IF( YAMLCPP_FOUND)
    SET( YAMLCPP_LIBRARIES ${YAMLCPP_LIBRARY})
    SET( YAMLCPP_INCLUDE_DIRS ${YAMLCPP_INCLUDE_DIR} ${YAMLCPP_INCLUDE_DIR}/yaml)
ELSE()
    SET( YAMLCPP_LIBRARIES)
    SET( YAMLCPP_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED( YAMLCPP_INCLUDE_DIR YAMLCPP_LIBRARY)
