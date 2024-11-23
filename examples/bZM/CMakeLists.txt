project(b612_ZM)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

set(TARGET bZM)
add_executable(bZM blorb.cpp branch.cpp dict.cpp iff.cpp io.cpp mathop.cpp meta.cpp memory.cpp objects.cpp options.cpp osdep.cpp patches.cpp process.cpp random.cpp screen.cpp sound.cpp speech.cpp stack.cpp stash.cpp unicode.cpp util.cpp zoom.cpp zterp.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/include_cxx_api)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/include_c_api)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/libs/Microsoft.CognitiveServices.Speech.core.lib)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/libs/Microsoft.CognitiveServices.Speech.extension.embedded.sr.lib)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/libs/Microsoft.CognitiveServices.Speech.extension.embedded.tts.lib)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/libs/Microsoft.CognitiveServices.Speech.extension.onnxruntime.lib)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/speech/libs/Microsoft.CognitiveServices.Speech.extension.telemetry.lib)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# end of CMakeLists.txt ...

