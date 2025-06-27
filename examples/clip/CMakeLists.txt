project(clip)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

set(TARGET cliplib)
add_library(
    cliplib OBJECT
    clip.cpp
    clip.h
    common-clip.cpp
    common-clip.h
)
add_library(cliplib_static STATIC $<TARGET_OBJECTS:cliplib>)
target_compile_features(${TARGET} PRIVATE cxx_std_20)
target_include_directories(${TARGET} PUBLIC ../..)

set(TARGET clipIndex)
add_executable(${TARGET} clipIndex.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_20)
target_link_libraries(${TARGET} PRIVATE common cliplib_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PUBLIC ../.. ${CMAKE_SOURCE_DIR}/hnswlib)

set(TARGET clipSrch)
add_executable(${TARGET} clipSrch.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_20)
target_link_libraries(${TARGET} PRIVATE common cliplib_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PUBLIC ../.. ${CMAKE_SOURCE_DIR}/hnswlib)
