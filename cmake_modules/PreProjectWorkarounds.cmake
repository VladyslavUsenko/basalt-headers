# workarounds that need to be applied before the `project(...)` call

# macOS < 10.14 requires that we set CMAKE_OSX_DEPLOYMENT_TARGET to
# 10.14 before `project(...)` is called, otherwise clang thinks
# certail C++17 STL features such as std::visit are not available,
# even if we use a recent libc++ from homebrew that has these
# features. Moreover, unless the compiler is specified, use clang from
# homebrew, since Apple's is too old.
if(APPLE)
  # Note: It is implicitly assumed that we use clang and libc++ from
  # brewed llvm on macOS < 10.14, since Apple's clang on those systems
  # is too old.

  # Note: CMAKE_SYSTEM_VERSION doesn't work before `project(...)`
  execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE _macos_version)
  string(REGEX REPLACE "\n$" "" _macos_version "${_macos_version}")
  if (_macos_version VERSION_LESS 10.14.0)
    message(STATUS "Detected macOS version '${_macos_version}', which is earlier than macOS 10.14 Mojave. Applying workarounds for clang and libc++...")

    # Ensure libc++ enables all features.
    # See: https://stackoverflow.com/a/53868971/1813258
    # See: https://stackoverflow.com/a/53887048/1813258
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")
    message(STATUS "... setting deployment target to '${CMAKE_OSX_DEPLOYMENT_TARGET}' to trick libc++ into not disabling some features (like std::visit)")

    # On macOS < 10.14, we need to ensure we use brewed clang.
    if (NOT CMAKE_C_COMPILER AND NOT CMAKE_CXX_COMPILER)
      set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
      set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
      message(STATUS "... compiler not specified; setting to brewed clang '${CMAKE_C_COMPILER}' and '${CMAKE_CXX_COMPILER}'")
    else()
      message(STATUS "... not setting compiler; already set to '${CMAKE_C_COMPILER}' and '${CMAKE_CXX_COMPILER}'")
    endif()
  else()
    message(STATUS "Detected macOS version '${_macos_version}', which is newer or equal to macOS 10.14 Mojave. Not applying workarounds.")
  endif()
endif()

