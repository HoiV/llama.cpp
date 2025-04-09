project(slm-infer)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

set(TARGET slm-infer)

add_library(
    slm-infer OBJECT
    dll/llm-infer.cpp
    dll/llm-infer.h
    )

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
#add_library(llm-infer_static STATIC $<TARGET_OBJECTS:slm-infer>)
add_library(llm-infer SHARED $<TARGET_OBJECTS:slm-infer>)
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${TARGET} PUBLIC ../..)
target_include_directories(${TARGET} PUBLIC ../../common)
target_include_directories(${TARGET} PRIVATE .)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include/hnswlib)
target_compile_definitions(${TARGET} PRIVATE LLM_INFER_SHARED LLM_INFER_DLL)
target_link_libraries(llm-infer PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS llm-infer LIBRARY)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET buildrag-dll)
add_executable(${TARGET} ragbuild/ragbuild.cpp ragbuild/zip_file.cpp ragbuild/zip_file.h ragbuild/dtt.h)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE .)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/dll)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include/hnswlib)
target_include_directories(${TARGET} PUBLIC ../../common)
target_link_libraries(${TARGET} PRIVATE common llama llm-infer ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(TARGET ragBuilder-dll)
add_executable(${TARGET} ragbuild/RAGbuilder/llm.cpp ragbuild/RAGbuilder/RAGbuilder.cpp ragbuild/RAGbuilder/RAGHelper.cpp ragbuild/RAGbuilder/Debug.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE .)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/dll)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include/hnswlib)
target_include_directories(${TARGET} PUBLIC ../../common)
target_link_libraries(${TARGET} PRIVATE common llama llm-infer ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(TARGET server-dll)
add_executable(${TARGET} server/server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE .)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/dll)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/llm-infer/include/hnswlib)
target_include_directories(${TARGET} PUBLIC ../../common)
target_link_libraries(${TARGET} PRIVATE common llama llm-infer ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# end of CMakeLists.txt ...

