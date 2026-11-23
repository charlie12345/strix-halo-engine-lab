# GLM-5.3 Flash image tower on the shared vision stack: the family adapter (strix_glm5_vision), its host test and the
# host tool that turns an image into an embed rows file (strix-glm5-vision; FP32 reference tower, no GPU).
if(TARGET strix_vision_gguf AND TARGET strix_vision_preprocess AND TARGET strix_model_data)
    add_library(strix_glm5_vision STATIC "${CMAKE_SOURCE_DIR}/src/glm/vision/glm_vision.cpp")
    target_include_directories(strix_glm5_vision PUBLIC "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(strix_glm5_vision PUBLIC strix_vision_gguf strix_vision_preprocess strix_model_data)
    if(MSVC)
        target_compile_definitions(strix_glm5_vision PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        # Rows are cache content and both ranks of a pair splice them: no fused multiply-add anywhere in the adapter.
        target_compile_options(strix_glm5_vision PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    strix_target_warnings(strix_glm5_vision)

    add_executable(strix-glm5-vision "${CMAKE_SOURCE_DIR}/apps/glm_vision_main.cpp")
    target_link_libraries(strix-glm5-vision PRIVATE strix_glm5_vision)
    if(MSVC)
        target_compile_definitions(strix-glm5-vision PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_target_warnings(strix-glm5-vision)
endif()

strix_servespec_host_test(NAME glm5-vision
    SOURCES tests/glm_vision_tests.cpp src/session/embed_rows.cpp
    LIBS strix_glm5_vision)
if(TARGET glm5-vision)
    if(NOT MSVC)
        target_compile_options(glm5-vision PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    set_tests_properties(glm5-vision PROPERTIES LABELS "cpu;glm")
endif()
