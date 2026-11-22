if(BUILD_TESTING)
    add_executable(strix-b7-screened-head-tests tests/cuda_core/screened_head_tests.cpp)
    target_include_directories(strix-b7-screened-head-tests PRIVATE include src)
    target_compile_features(strix-b7-screened-head-tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(strix-b7-screened-head-tests PRIVATE /W4 /WX /fp:strict)
    else()
        target_compile_options(strix-b7-screened-head-tests PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-b7-screened-head-tests COMMAND strix-b7-screened-head-tests)
    set_tests_properties(strix-b7-screened-head-tests PROPERTIES LABELS "cpu;head")
endif()

option(STRIX_B7_CUDA_FIXTURE "Compile the B7 screened head conformance fixture for sm_120a" OFF)
if(STRIX_B7_CUDA_FIXTURE)
    enable_language(CUDA)
    add_executable(strix-b7-screened-head-fixture tests/cuda_core/screened_head_fixture.cu
        src/cuda/core/dense/dense_mma.cu src/cuda/core/dense/nvfp4.cu)
    target_include_directories(strix-b7-screened-head-fixture PRIVATE include src)
    set_target_properties(strix-b7-screened-head-fixture PROPERTIES
        CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strix-b7-screened-head-fixture PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
    if(MSVC)
        target_compile_options(strix-b7-screened-head-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8,/fp:strict>)
    else()
        target_compile_options(strix-b7-screened-head-fixture PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math,-ffp-contract=off>)
    endif()
    # Device execution is an explicit operator step, outside CTest.
endif()
