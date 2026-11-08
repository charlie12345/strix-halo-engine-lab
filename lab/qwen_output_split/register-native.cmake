# Explicit root project hook; ordinary root configuration does not include it.
cmake_language(EVAL CODE "cmake_language(DEFER CALL include [[${CMAKE_CURRENT_LIST_DIR}/native-target.cmake]])")
