# Provided embedding rows in the Inkling decoder (prefill_embedded): every HIP target that compiles the resident
# decoder links the shared splice kernel. Image input of the Inkling CLI (--messages with image parts): the vision
# adapter on the device tower through the shared serial media input. (After vise2e_media.cmake, which defines
# strix_vision_api_media.)
if(TARGET strix_embed_splice_hip)
    foreach(strix_ink_target strix-inkling2-mini-hip strix_inkling2_mini_serve_device
            strix-hip-inkling2-mini-verify-rows-test strix-hip-inkling2-mini-dspark-test)
        if(TARGET ${strix_ink_target})
            target_link_libraries(${strix_ink_target} PRIVATE strix_embed_splice_hip)
        endif()
    endforeach()
endif()
if(TARGET strix-inkling2-mini-hip AND TARGET strix_inkling2_vision AND TARGET strix_vision_api_media AND
   TARGET strix_vision_encoder_hip)
    target_link_libraries(strix-inkling2-mini-hip PRIVATE strix_inkling2_vision strix_vision_api_media
        strix_vision_encoder_hip)
    target_compile_definitions(strix-inkling2-mini-hip PRIVATE STRIX_INKLING2_CLI_VISION=1)
endif()
