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
# The execution owner has no dependency on an HTTP implementation. Replay's
# observed queue and production's queue both link this same host owner archive.
add_library(strix_session_owner STATIC "${_strix_session_root}/src/serve/session_runtime.cpp")
target_link_libraries(strix_session_owner PUBLIC strix_session)
strix_session_warnings(strix_session_owner)

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
function(strix_session_add_runtime)
    add_library(strix_session_runtime INTERFACE)
    target_link_libraries(strix_session_runtime INTERFACE strix_session_owner strix_serve strix_model_registry)
    add_library(strix_session_shell STATIC "${_strix_session_root}/src/serve/run.cpp")
    target_link_libraries(strix_session_shell PUBLIC strix_session_runtime)
    strix_session_warnings(strix_session_shell)
    add_executable(strix-serve "${_strix_session_root}/apps/strix_serve_main.cpp")
    target_link_libraries(strix-serve PRIVATE strix_session_shell)
    strix_session_warnings(strix-serve)
    if(BUILD_TESTING)
        foreach(test runtime http_replay registry turn_cache)
            add_executable(session_${test}_tests "${_strix_session_root}/tests/session_${test}_tests.cpp")
            target_link_libraries(session_${test}_tests PRIVATE strix_session_shell)
            strix_session_warnings(session_${test}_tests)
            strix_session_test_style(session_${test}_tests)
            add_test(NAME session_${test} COMMAND session_${test}_tests "${_strix_session_root}")
            set_tests_properties(session_${test} PROPERTIES LABELS "cpu;session" TIMEOUT 30)
        endforeach()
    endif()
    if(BUILD_TESTING AND NOT TARGET http_shutdown_tests)
        add_subdirectory("${_strix_session_root}/tests/http_shutdown" "${CMAKE_BINARY_DIR}/http-shutdown")
    endif()
endfunction()


# Explicit optional archive reference without a runtime/family dependency cycle.
function(strix_session_link_services)
    if(TARGET strix_gemma4_session_service AND TARGET strix_session_shell)
        target_compile_definitions(strix_session_shell PRIVATE STRIX_SERVE_GEMMA4=1)
        target_link_libraries(strix_session_shell PRIVATE strix_gemma4_session_service)
        if(TARGET session_registry_tests)
            target_compile_definitions(session_registry_tests PRIVATE STRIX_EXPECT_GEMMA4_SERVICE=1)
        endif()
        # A device-backed service makes HIP the link language of every shell
        # consumer; these targets were declared before the HIP target list.
        if(STRIX_ENABLE_HIP_RUNTIME AND DEFINED STRIX_HIP_ARCHS)
            foreach(_t strix_session_shell strix-serve session_runtime_tests session_http_replay_tests
                    session_registry_tests session_turn_cache_tests)
                if(TARGET ${_t})
                    set_target_properties(${_t} PROPERTIES HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
                endif()
            endforeach()
        endif()
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL strix_session_link_services)

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
