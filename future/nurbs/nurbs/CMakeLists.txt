
INCLUDE_DIRECTORIES(	../
                        ${Boost_INCLUDE_DIR}
                        )

# Fake lib
FILE(GLOB_RECURSE NURBS_HEADERS ./*.hpp)
SET( NURBS_SOURCES empty.cpp)

SET( NURBS_ALL   ${NURBS_SOURCES}
                 ${NURBS_HEADERS}
                 )

ADD_LIBRARY( nurbs_empty SHARED ${NURBS_ALL})
