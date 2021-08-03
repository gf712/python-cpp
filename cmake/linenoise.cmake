include(FetchContent)


FetchContent_Declare(linenoise_src
    GIT_REPOSITORY    git@github.com:antirez/linenoise.git
    GIT_TAG           master
)

FetchContent_MakeAvailable(linenoise_src)

FetchContent_GetProperties(linenoise_src)
if(NOT linenoise_src_POPULATED)
  FetchContent_Populate(linenoise_src)
endif()

add_library(linenoise ${linenoise_src_SOURCE_DIR}/linenoise.c)
target_include_directories(linenoise PUBLIC ${linenoise_src_SOURCE_DIR})