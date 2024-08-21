project(mjZ)

set(MOJOZORK_STANDALONE_DEFAULT ON)
set(MOJOZORK_MULTIZORK_DEFAULT ON)
set(MOJOZORK_LIBRETRO_DEFAULT ON)
set(minZ_DEFAULT ON)

if(MSVC)   # the daemon is pretty Unixy.
    set(MOJOZORK_MULTIZORK_DEFAULT OFF)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(ANDROID TRUE)
endif()

option(MOJOZORK_STANDALONE "Build the MojoZork standalone app" ${MOJOZORK_STANDALONE_DEFAULT})
option(MOJOZORK_MULTIZORK "Build the Multizork server" ${MOJOZORK_MULTIZORK_DEFAULT})
option(minZ_STANDALONE "Build the minZ app" ${minZ_DEFAULT})

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
else()  # This assumes you have GCC or Clang at the moment.
    add_definitions(-Wall)
endif()

if(MOJOZORK_STANDALONE)
    set(TARGET mjZ)
    add_executable(mjZ mojozork.c)
endif()

if(MOJOZORK_MULTIZORK)
    add_executable(multiZ multizorkd.c)
    target_link_libraries(multiZ -lsqlite3)
endif()

if(minZ_STANDALONE)
    set(TARGET minZ)
    add_executable(minZ minZ.c)
endif()

# end of CMakeLists.txt ...

