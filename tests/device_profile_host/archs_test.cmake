include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/StrixHipArchs.cmake")
if(NOT STRIX_HIP_ARCHS STREQUAL EXPECTED)
    message(FATAL_ERROR "Expected '${EXPECTED}', got '${STRIX_HIP_ARCHS}'")
endif()
