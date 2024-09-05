llama_print_timings:        load time =   134.84 ms
llama_print_timings:      sample time =    12.25 ms /    32 runs   (    0.38 ms per token)
llama_print_timings: prompt eval time =   925.78 ms /    24 tokens (   38.57 ms per token)
llama_print_timings:        eval time =  4178.32 ms /    31 runs   (  134.78 ms per token)
llama_print_timings:       total time =  5211.06 ms

load time: loading model file
sample time: choosing the next likely token.
prompt eval time: how long it took to process the prompt/file by LLaMa before generating new text.
eval time: how long it took to generate the output (until [end of text] or the user set limit).
total: all together
When it says 38.57 ms per token, it means it evaluated 24 tokens in 925.78 ms. Lower is better.

Another common metric is tokens per second which is used in other ML projects than llama.cpp. You can get that by calculating 1000/38.57 = 25.93 t/s. Higher is better.

Why is eval time slower than prompt eval time?

Because we can only predict the single next token, that means that after the sampler chooses the next token the model has to be run again but with a batch size of 1. In the prompt eval phase, the model can evaluate large batches (512 max) meaning less overhead and more efficiency, especially with BLAS or GPU.

EDIT: "sample time" is not tokenization time, it is actually the "sampling" time, that means running the RNG, sorting and filtering candidates, etc.

=============================================================================

Up to PR 10678775 - Remove remaining multiply matrix init serialized code (fork from hv/matmul)

10689768 - use _mm256_dpbusd to speed up the summation of 16-byte chunks
10710959 - speed up q4 dot product
10712792 - dispatch unary functions directly via a function table

commit 134ee5bf81cfcec2fe8bbae8d2e88da567e20ad2 (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Mon May 20 16:50:58 2024 -0700

    Up to PR 10712792: dispatch unary functions directly via a function table.

10724430 - start conversion of dispatch to directly dispatch rather than call function

commit 27e2dce45093f8ec8d80297d7e7558d11b6184af (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Tue May 21 00:23:23 2024 -0700

    PR 10724430 - start conversion of dispatch to directly dispatch rather than call function

10738692 - Next tranche of changes to convert tensor dispatch into a direct call

commit a0d8a3f0e42c0231c2d4a86effd0183a53297292 (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Fri May 24 20:41:04 2024 -0700

    PR 10738692 - Next tranche of changes to convert tensor dispatch into a direct call

10748893 - ai complete conversion to direct dispatch of tensor functions

commit d191d87da84393d07a62ef959f426d8c4e2d301b (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Fri May 24 21:53:33 2024 -0700

    PR 10748893 - ai complete conversion to direct dispatch of tensor functions

10790116 - ai - convert ggml_compute_forward_sub_f32 to parallel execution

commit 5eef4f449f55f0c674627f44ecb028ce3d121375 (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Fri May 24 22:39:03 2024 -0700

    PR 10790116 - ai - convert ggml_compute_forward_sub_f32 to parallel execution

PR 10800462 - ai - vectorize ggml_fp16_to_fp32 and simplify asserts.
PR 10807564 - ai parallelize ggml_compute_forward_get_rows

commit 76d7f1c1f05fee4cba1b3be8e221120e8a07c791 (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Fri May 24 23:59:53 2024 -0700

    PR 10800462 - ai - vectorize ggml_fp16_to_fp32 and simplify asserts.
    PR 10807564 - ai parallelize ggml_compute_forward_get_rows

Pickup changes up to PR pullrequest/11177694
    PR 11177694: move tensor init and finalize processing and execution synchronization out of
the tensor dispatcher loop to the target tensor computation. Most tensors have neither an
init or finalize requirement and this removes branch and test code from a critical path to
the actually tensor code. This change also enables both init and finalize code to run in
parallel which was previously not available (a special case was in place for mul_mat).

******************* CHANGE notice for latest change in llama.cpp *******************
To get static libs: e:\Xbox-B612\src\llama.cpp>cmake .. -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON
The default is to generate ggml.dll and llama.dll
************************************************************************************

D:\llama.cpp\llama.cpf_clang\build>cmake .. -T CLangCL -DBUILD_SHARED_LIBS=ON -DGGML_STATIC=OFF
-- Building for: Visual Studio 17 2022
-- Selecting Windows SDK version 10.0.22621.0 to target Windows 10.0.22631.
-- The C compiler identification is Clang 17.0.3 with MSVC-like command-line
-- The CXX compiler identification is Clang 17.0.3 with MSVC-like command-line
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.31.0.vfs.0.1")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - not found
-- Found Threads: TRUE
-- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
-- Could NOT find OpenMP (missing: OpenMP_C_FOUND OpenMP_CXX_FOUND)
CMake Warning at CMakeLists.txt:312 (message):
  OpenMP not found


-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with LLAMA_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: AMD64
-- CMAKE_GENERATOR_PLATFORM:
-- x86 detected
-- Performing Test HAS_AVX_1
-- Performing Test HAS_AVX_1 - Failed
-- Performing Test HAS_AVX_2
-- Performing Test HAS_AVX_2 - Success
-- Performing Test HAS_AVX2_1
-- Performing Test HAS_AVX2_1 - Failed
-- Performing Test HAS_AVX2_2
-- Performing Test HAS_AVX2_2 - Success
-- Performing Test HAS_FMA_1
-- Performing Test HAS_FMA_1 - Failed
-- Performing Test HAS_FMA_2
-- Performing Test HAS_FMA_2 - Success
-- Performing Test HAS_AVX512_1
-- Performing Test HAS_AVX512_1 - Failed
-- Performing Test HAS_AVX512_2
-- Performing Test HAS_AVX512_2 - Success
-- Configuring done (17.2s)
-- Generating done (0.4s)
CMake Warning:
  Manually-specified variables were not used by the project:

    GGML_STATIC


-- Build files have been written to: D:/llama.cpp/llama.cpf_clang/build

D:\llama.cpp\llama.cpf_clang\build>cmake --build . -j 12 --config RelWithDebInfo --target llama-bench
MSBuild version 17.11.2+c078802d4 for .NET Framework

  1>Checking Build System
  Generating build details from Git
  -- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.31.0.vfs.0.1")
  Building Custom Rule D:/llama.cpp/llama.cpf_clang/common/CMakeLists.txt
  build_info.vcxproj -> D:\llama.cpp\llama.cpf_clang\build\common\build_info.dir\RelWithDebInfo\build_info.lib
...
D:\llama.cpp\llama.cpf_clang\unicode.cpp(18,20): warning : unused function 'unicode_cpts_to_utf8' [-Wunused-function] [
D:\llama.cpp\llama.cpf_clang\build\llama.vcxproj]
  Auto build dll exports
  llama.vcxproj -> D:\llama.cpp\llama.cpf_clang\build\bin\RelWithDebInfo\llama.dll
  Building Custom Rule D:/llama.cpp/llama.cpf_clang/common/CMakeLists.txt
D:\llama.cpp\llama.cpf_clang\common\grammar-parser.cpp(413,17): warning : unused function 'print_rule_binary' [-Wunused
-function] [D:\llama.cpp\llama.cpf_clang\build\common\common.vcxproj]
D:\llama.cpp\llama.cpf_clang\common\json-schema-to-grammar.cpp(122,20): warning : unused function 'repeat' [-Wunused-fu
nction] [D:\llama.cpp\llama.cpf_clang\build\common\common.vcxproj]
  common.vcxproj -> D:\llama.cpp\llama.cpf_clang\build\common\RelWithDebInfo\common.lib
  Building Custom Rule D:/llama.cpp/llama.cpf_clang/examples/llama-bench/CMakeLists.txt
  llama-bench.vcxproj -> D:\llama.cpp\llama.cpf_clang\build\bin\RelWithDebInfo\llama-bench.exe

D:\llama.cpp\llama.cpf_clang\build>dir bin\RelWithDebInfo
 Volume in drive D is New Volume
 Volume Serial Number is 8030-7123

 Directory of D:\llama.cpp\llama.cpf_clang\build\bin\RelWithDebInfo

09/03/2024  12:09 PM    <DIR>          .
09/03/2024  12:08 PM    <DIR>          ..
09/03/2024  12:09 PM           973,312 llama-bench.exe
09/03/2024  12:09 PM         8,359,936 llama-bench.pdb
09/03/2024  12:09 PM         2,469,376 llama.dll
09/03/2024  12:09 PM        10,575,872 llama.pdb
               4 File(s)     22,378,496 bytes
               2 Dir(s)  1,397,066,555,392 bytes free
