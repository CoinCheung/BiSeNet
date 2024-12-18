


add_library (custom_plugin SHARED argmax_plugin.cu)
target_compile_features (custom_plugin PRIVATE cuda_std_14)
target_include_directories (custom_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

