# Add project_options v0.25.2
# https://github.com/aminya/project_options
# Change the version in the following URL to update the package (watch the releases of the repository for future updates)
include(FetchContent)
FetchContent_Declare(_project_options
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/aminya/project_options/archive/refs/tags/v0.25.2.zip
)
FetchContent_MakeAvailable(_project_options)
include(${_project_options_SOURCE_DIR}/Index.cmake)