cmake_minimum_required( VERSION 3.12.0 )
project (stencil)
find_package( Threads REQUIRED )
find_package( OpenMP 4.0 REQUIRED )

# Lyra Cmdline
find_path(BFGROUP_LYRA_INCLUDE_DIRS "lyra/arg.hpp")

add_executable( stencil-benchmark-knl stencil-benchmark.cpp stencil-parallel.cpp)
target_include_directories( stencil-benchmark-knl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries( stencil-benchmark-knl
	PRIVATE
  OpenMP::OpenMP_CXX
  Threads::Threads
  )
target_compile_features( stencil-benchmark-knl 
	PRIVATE
  cxx_std_17
  )
if( CMAKE_CXX_COMPILER MATCHES "icpc.*$" ) # Intel compiler
	target_compile_options(stencil-benchmark-knl 
		PRIVATE
		-march=knl
		-vecabi=cmdtarget
		-qopt-zmm-usage=high
		-qopt-report=5
		-qopt-report-phase=vec
		-save-temps
		)
else()
	target_compile_options( stencil-benchmark-knl 
		PRIVATE
		-march=knl 
		-funroll-loops
		-ffast-math
		-m64
		)
endif()

add_executable( stencil-benchmark-skl stencil-benchmark.cpp stencil-parallel.cpp)
target_include_directories( stencil-benchmark-skl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries( stencil-benchmark-skl
	PRIVATE
  OpenMP::OpenMP_CXX
  Threads::Threads
  )
target_compile_features( stencil-benchmark-skl 
	PRIVATE
  cxx_std_17
  )
if( CMAKE_CXX_COMPILER MATCHES "icpc.*$" ) # Intel compiler
	target_compile_options(stencil-benchmark-skl 
		PRIVATE
		-march=skylake
		-vecabi=cmdtarget
		-qopt-zmm-usage=high
		-qopt-report=5
		-qopt-report-phase=vec
		-save-temps
		)
else()
	target_compile_options( stencil-benchmark-skl 
		PRIVATE
		-march=skylake 
		-funroll-loops
		-ffast-math
		-mfpmath=sse
		)
endif()

