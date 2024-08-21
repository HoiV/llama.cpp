project(bocfel_ZM)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

set(TARGET bZM)
add_executable(bZM blorb.cpp branch.cpp dict.cpp iff.cpp io.cpp mathop.cpp meta.cpp memory.cpp objects.cpp options.cpp osdep.cpp patches.cpp process.cpp random.cpp screen.cpp sound.cpp stack.cpp stash.cpp unicode.cpp util.cpp zoom.cpp zterp.cpp)

# end of CMakeLists.txt ...

