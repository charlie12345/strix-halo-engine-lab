# Host session resources and storage are independent of accelerator languages
# and strix_core. Standalone test projects retain their own entry points.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/StrixStateCache.cmake")
get_filename_component(_strix_session_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Host replay projects retain a configure-time copy of this production source.
# Regenerate that copy after queue API changes instead of linking a stale copy
# against current session headers. No frozen expectations are regenerated.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_strix_session_root}/src/serve/engine.cpp")
function(strix_session_warnings target)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(${target})
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /EHsc /permissive-)
    endif()
endfunction()

# Host fixtures are written densely; keep their style out of the error set
# while every other warning stays an error.
function(strix_session_test_style target)
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -Wno-misleading-indentation -Wno-unused-function)
    endif()
endfunction()

add_library(strix_session STATIC
    "${_strix_session_root}/src/session/backend_cpu.cpp"
    "${_strix_session_root}/src/session/kv.cpp"
    "${_strix_session_root}/src/session/prefix_cache.cpp"
    "${_strix_session_root}/src/session/parallel_job.cpp")
target_include_directories(strix_session PUBLIC "${_strix_session_root}/include")
target_compile_features(strix_session PUBLIC cxx_std_20)
target_link_libraries(strix_session PUBLIC strix_state_cache)
set_target_properties(strix_session PROPERTIES CXX_EXTENSIONS OFF)
strix_session_warnings(strix_session)

# Central host contract/test registration. Add future cost/config (PR2),
# protocol (PR3) and runner (PR5) implementations to strix_session above and
# their real tests to this list when present. Missing future APIs are pending;
# do not register placeholder executables or empty passing tests.
function(strix_session_add_contract_tests)
    if(NOT BUILD_TESTING)
        return()
    endif()
    foreach(test engine_contract backend kv prefix_cache recurrent fanout parallel_job cache_owner)
        add_executable(session_${test}_tests "${_strix_session_root}/tests/session_${test}_tests.cpp")
        target_link_libraries(session_${test}_tests PRIVATE strix_session)
        strix_session_warnings(session_${test}_tests)
        strix_session_test_style(session_${test}_tests)
        if(MSVC)
            target_compile_options(session_${test}_tests PRIVATE /UNDEBUG)
        else()
            target_compile_options(session_${test}_tests PRIVATE -UNDEBUG)
        endif()
        add_test(NAME session_${test} COMMAND session_${test}_tests)
        set_tests_properties(session_${test} PROPERTIES LABELS "cpu;session" TIMEOUT 30)
    endforeach()
endfunction()
strix_session_add_contract_tests()

# The runtime paths and optional existing launcher target are defined later by
# the root build. No dependency in the reverse direction through strix_core.
function(strix_session_add_hip)
    if(NOT STRIX_ENABLE_HIP_RUNTIME OR STRIX_CUDA_QWEN_ONLY)
        return()
    endif()
    add_library(strix_session_hip STATIC "${_strix_session_root}/src/session/backend_hip.cpp")
    target_link_libraries(strix_session_hip PUBLIC strix_session PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_session_hip SYSTEM PRIVATE "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_definitions(strix_session_hip PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_features(strix_session_hip PUBLIC cxx_std_20)
    if(TARGET strix_ds41_trellis_hip)
        target_link_libraries(strix_session_hip PRIVATE strix_ds41_trellis_hip)
        target_compile_definitions(strix_session_hip PRIVATE STRIX_SESSION_HIP_ADD=1)
    endif()
    strix_session_warnings(strix_session_hip)
endfunction()
cmake_language(DEFER CALL strix_session_add_hip)
