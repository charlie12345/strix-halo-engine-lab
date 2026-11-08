cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/StrixHipArchs.cmake")

project(strix_moe_experts_lab LANGUAGES CXX HIP)

find_package(hip REQUIRED CONFIG)

add_executable(strix-moe-experts moe_experts.hip)
set_target_properties(strix-moe-experts PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}"
)
target_link_libraries(strix-moe-experts PRIVATE hip::host)

if(MSVC)
    target_compile_options(strix-moe-experts PRIVATE /W4)
else()
    target_compile_options(
        strix-moe-experts PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()
