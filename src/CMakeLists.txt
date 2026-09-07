# ----------------------------------------------------------
#  Mitsuba library
# ----------------------------------------------------------

# Some parts of Mitsuba use the TSL hash map class
include_directories(../ext/nanobind/ext/robin_map/include)

add_subdirectory(core)
add_subdirectory(render)

add_library(mitsuba SHARED)
target_link_libraries(mitsuba PUBLIC mitsuba-core)
target_link_libraries(mitsuba PUBLIC mitsuba-render)
set_target_properties(mitsuba PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

# Scalar CPU bridge to the SFWN representation in the sibling research clone.
set(SFWN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../point_cloud_utils/cpp")
find_package(CGAL REQUIRED)
find_package(OpenMP REQUIRED)
target_sources(mitsuba PRIVATE
  sfwn/sfwn.cpp
  ${SFWN_SOURCE_DIR}/third_party/miniply.cpp
)
target_include_directories(mitsuba PRIVATE
  ${SFWN_SOURCE_DIR}/include
  ${SFWN_SOURCE_DIR}/third_party
)
target_link_libraries(mitsuba PRIVATE CGAL::CGAL OpenMP::OpenMP_CXX)
target_compile_definitions(mitsuba PRIVATE MI_BUILD_MODULE=MI_MODULE_LIB)
set_property(TARGET mitsuba PROPERTY CXX_STANDARD 20)
set_source_files_properties(
  ${SFWN_SOURCE_DIR}/third_party/miniply.cpp
  PROPERTIES COMPILE_OPTIONS "-w"
)

add_executable(sfwn_probe sfwn/probe.cpp)
target_link_libraries(sfwn_probe PRIVATE mitsuba)
set_property(TARGET sfwn_probe PROPERTY CXX_STANDARD 20)

if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU)$")
  # Treat undefined symbols as errors when linking
  target_link_libraries(mitsuba PRIVATE -Wl,--no-undefined)
  # Silence an incorrect warning message in GCC LTO builds
  target_link_options(mitsuba PRIVATE -Walloc-size-larger-than=18446744073709551615)
endif()

# ----------------------------------------------------------
#  Mitsuba executable
# ----------------------------------------------------------

add_subdirectory(mitsuba)

# ----------------------------------------------------------
#  Plugins
# ----------------------------------------------------------

add_subdirectory(bsdfs)
add_subdirectory(emitters)
add_subdirectory(films)
add_subdirectory(integrators)
add_subdirectory(media)
add_subdirectory(phase)
add_subdirectory(rfilters)
add_subdirectory(samplers)
add_subdirectory(sensors)
add_subdirectory(shapes)
add_subdirectory(spectra)
add_subdirectory(textures)
add_subdirectory(volumes)
set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)

# ----------------------------------------------------------
#  Python bindings and extensions
# ----------------------------------------------------------

if (MI_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
