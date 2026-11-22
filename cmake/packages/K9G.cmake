if(BUILD_TESTING)
    foreach(form IN ITEMS decode decode-step)
        string(REPLACE "-" "_" source_form ${form})
        set(target strix-k9g-${form}-plan-test)
        add_executable(${target} tests/cuda_core/recur_${source_form}_plan_test.cpp)
        target_include_directories(${target} PRIVATE include src)
        target_compile_features(${target} PRIVATE cxx_std_20)
        strix_target_warnings(${target})
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "cpu;K9G")
    endforeach()
endif()

option(STRIX_K9G_CUDA_FIXTURE "Compile the gfx1151 GDN decode comparisons for sm_120a" OFF)
if(STRIX_K9G_CUDA_FIXTURE)
    enable_language(CUDA)
    foreach(form IN ITEMS decode decode-step)
        string(REPLACE "-" "_" source_form ${form})
        set(target strix-k9g-${form}-fixture)
        add_executable(${target}
            tests/cuda_core/recur_${source_form}_fixture.cu
            src/cuda/core/recur_decode.cu
            src/cuda/core/amd_math_tables.cu)
        target_include_directories(${target} PRIVATE include src)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
        target_compile_options(${target} PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
        if(MSVC)
            target_compile_options(${target} PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
        endif()
    endforeach()
    # The operator runs these fixtures with TABLES_DIR. CTest runs only host tests.
endif()
