e:\Xbox-B612\llama.master>git clone https://github.com/hoiv/llama.cpp .
Cloning into '.'...
remote: Enumerating objects: 14309, done.
remote: Counting objects: 100% (14309/14309), done.
remote: Compressing objects: 100% (4159/4159), done.
remote: Total 14309 (delta 10075), reused 14188 (delta 10009), pack-reused 0R
Receiving objects: 100% (14309/14309), 19.67 MiB | 11.37 MiB/s, done.
Resolving deltas: 100% (10075/10075), done.

e:\Xbox-B612\llama.master>git reflog
557410b (HEAD -> master, origin/master, origin/HEAD, info) HEAD@{0}: clone: from https://github.com/hoiv/llama.cpp

e:\Xbox-B612\llama.master>git branch -a
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/hv/matmul
  remotes/origin/master

e:\Xbox-B612\llama.master>git show-branch --all
! [info] llama : greatly reduce output buffer memory usage (#6122)
 * [master] llama : greatly reduce output buffer memory usage (#6122)
  ! [origin/HEAD] llama : greatly reduce output buffer memory usage (#6122)
   ! [origin/hv/matmul] Complete the vectorization for quantize_row_q8_k()
    ! [origin/master] llama : greatly reduce output buffer memory usage (#6122)
-----
   +  [origin/hv/matmul] Complete the vectorization for quantize_row_q8_k()
   +  [origin/hv/matmul^] Update with vectorization of quantize_row_q8_k()
   +  [origin/hv/matmul~2] Fix ggml_vec_max_f32
   +  [origin/hv/matmul~3] Vectorize more operators Add profiling info summary
   +  [origin/hv/matmul~4] Update for -cfp support (custom prompts)
+*+++ [info] llama : greatly reduce output buffer memory usage (#6122)

=============================================================================

To build for ARM64 (Cadmus)

cmake --preset arm64-windows-llvm-release -D LLAMA_LLAMAFILE=OFF -D LLAMA_OPENMP=OFF -B build.clang.preset

Preset CMake variables:

  CMAKE_BUILD_TYPE="RelWithDebInfo"
  CMAKE_EXPORT_COMPILE_COMMANDS="ON"
  CMAKE_INSTALL_RPATH="$ORIGIN;$ORIGIN/.."
  CMAKE_TOOLCHAIN_FILE="C:/llama.cpp/llama.cpf_q4/cmake/arm64-windows-llvm.cmake"

-- The C compiler identification is Clang 17.0.3 with GNU-like command-line
-- The CXX compiler identification is Clang 17.0.3 with GNU-like command-line
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/ARM64/bin/clang.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/ARM64/bin/clang++.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.47.0.windows.1")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - no
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - not found
-- Found Threads: TRUE
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with LLAMA_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: arm64
-- ARM detected
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E
-- Performing Test COMPILER_SUPPORTS_FP16_FORMAT_I3E - Failed
-- Configuring done (3.8s)
-- Generating done (0.1s)
-- Build files have been written to: C:/llama.cpp/llama.cpf_q4/build.clang.preset

C:\llama.cpp\llama.cpf_q4>cd build.clang.preset

C:\llama.cpp\llama.cpf_q4\build.clang.preset>cmake --build . --target llama-bench kv
...
