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

INCLUDE_DIRECTORIES( ../../
                     ${Boost_INCLUDE_DIR}
                     )

# lib
FILE( GLOB_RECURSE RAMEN_HASH_HEADERS ./*.hpp)
SET( RAMEN_HASH_SOURCES  extern/MurmurHash3.cpp
                         murmur3_hash_fun.cpp
                         md5_hash_fun.cpp
                         )

SET( RAMEN_HASH_LIBS)
SET( RAMEN_HASH_ALL ${RAMEN_HASH_SOURCES}
                    ${RAMEN_HASH_HEADERS}
                    )

ADD_LIBRARY( ramen_hash SHARED ${RAMEN_HASH_ALL})
TARGET_LINK_LIBRARIES( ramen_hash ${RAMEN_HASH_LIBS})
