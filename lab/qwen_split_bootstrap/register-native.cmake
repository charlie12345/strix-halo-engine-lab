# Pass only on an explicit, opt-in project configure.
cmake_language(EVAL CODE "cmake_language(DEFER CALL include [[${CMAKE_CURRENT_LIST_DIR}/native-target.cmake]])")
