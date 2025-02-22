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

===================================================================================

Never edit master. Keep it in the state of the original upstream fork (github.com/ggerganov/llama.cpp.git).

1.	“Sync fork” to merge the master branch of the original upstream fork (github.com/ggerganov/llama.cpp.git) into our master branch 
        of the downstream GitHub fork (github.com/HoiV/llama.cpp.git).
2.	“git fetch” to merge the master branch of the downstream GitHub fork (github.com/HoiV/llama.cpp.git) to the master branch of the local clone.
3.	While on the topic branch, “git merge origin/master” to merge the master branch of the local clone to the specific topic branch.
4.	#3 will have conflicts that need to resolve.


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

================================================================================

OpenBLAS

C:\llama.cpp\llama.cpf\build.clang.openblas>cmake ..            -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DBLAS_LIBRARIES=c:\llama.cpp\openBLAS_rel\lib -DBLAS_INCLUDE_DIRS=c:\llama.cpp\openBLAS_rel\include
C:\llama.cpp\llama.cpf\build.clang.openblas>cmake .. -T CLangCL -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DBLAS_LIBRARIES=c:\llama.cpp\openBLAS_rel\lib -DBLAS_INCLUDE_DIRS=c:\llama.cpp\openBLAS_rel\include

C:\llama.cpp\llama.cpf\build.clang.openblas>copy c:\llama.cpp\openBLAS_rel\include\cblas.h ..\examples\kv-cache
C:\llama.cpp\llama.cpf\build.clang.openblas>copy c:\llama.cpp\openBLAS_rel\bin\libopenblas.dll bin\RelWithDebInfo

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   ../examples/kv-cache/CMakeLists.txt

set(TARGET kv)
add_executable(${TARGET} kv-cache.cpp slminfer.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(kv PUBLIC .)
target_include_directories(kv PUBLIC ../..)
target_include_directories(kv PUBLIC ../../common)
>>> target_link_libraries(${TARGET} PRIVATE llama ${BLAS_LIBRARIES}/libopenblas.lib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

        modified:   ../examples/kv-cache/kv-cache.h

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

>>> #ifdef GGML_USE_BLAS
>>> #include <cblas.h>
>>> #endif

#include <cmath>
#include <cstdio>

        modified:   ../examples/llama-bench/CMakeLists.txt

set(TARGET llama-bench)
add_executable(${TARGET} llama-bench.cpp)
install(TARGETS ${TARGET} RUNTIME)
>>> target_link_libraries(${TARGET} PRIVATE common llama  ${BLAS_LIBRARIES}/libopenblas.lib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

        modified:   ../examples/llama-bench/llama-bench.cpp

    // select openmp if specified
    if (params.openmp) {
>>>        // ggml_select_omp();
    }


Untracked files:
  (use "git add <file>..." to include in what will be committed)
        ../examples/kv-cache/cblas.h
        ../examples/kv-cache/openblas_config.h

================================================================================

find_package()

OpenMP: 
 Directory of d:\cmake\share\cmake-3.24\Modules
07/29/2022  03:13 PM            26,803 FindOpenMP.cmake
- or -
 Directory of C:\Program Files\CMake\share\cmake-3.30\Modules
06/17/2024  03:15 PM            29,034 FindOpenMP.cmake

RyzenAI:
 Directory of c:\ProgramData\anaconda3\envs\ryzenai-transformers\Lib\cmake\ryzenai
10/05/2024  11:54 AM    <DIR>          .
10/04/2024  02:47 PM    <DIR>          ..
10/04/2024  01:02 PM             1,033 RyzenAIConfig.cmake
10/05/2024  11:53 AM             4,596 RyzenAIConfigTargets.cmake
10/05/2024  11:53 AM             1,904 RyzenAIConfigVersion.cmake
               3 File(s)          7,533 bytes
               2 Dir(s)  143,886,049,280 bytes free

=================================================================================

REM Prerequisites (files and environment) for building llama.cpp LLAMA_RYZENAI=ON

C:\llama.cpp\Ryzen>git reflog (on Strix system - ie. 10.159.22.81)
e6086a1 (HEAD -> main, origin/main, origin/HEAD) HEAD@{0}: clone: from https://github.com/hoivb612/RyzenAI-SW

C:\llama.cpp\Ryzen\example\transformers\ops\cpp>cmake --install build\ --config Release
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/lib/_RyzenAI.cp311-win_amd64.pyd
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/ryzenai.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/ops/qlinear_2
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/ops/qlinear_2/py_qlinear_2.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/ops/qlinear_2/qlinear_2.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/buffer_ops.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/dpu_kernel_metadata.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/dtype_utils.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/instruction_registry.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/logging.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/matrix_formatting.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/ml_params.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/stats.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/super_instr.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/threadpool.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/utils.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/wgt_matrix.h
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/include/ryzenai/utils/xrt_context.hpp
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/lib/cmake/ryzenai/RyzenAIConfigTargets.cmake
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/lib/cmake/ryzenai/RyzenAIConfig.cmake
-- Up-to-date: C:/ProgramData/anaconda3/envs/ryzenai-transformers/lib/cmake/ryzenai/RyzenAIConfigVersion.cmake

C:\llama.cpp\Ryzen\example\transformers\ops\cpp>set
ALLUSERSPROFILE=C:\ProgramData
APPDATA=C:\Users\Administrator\AppData\Roaming
CLIENTNAME=FUSIONAI1
CommandPromptType=Native
CommonProgramFiles=C:\Program Files\Common Files
CommonProgramFiles(x86)=C:\Program Files (x86)\Common Files
CommonProgramW6432=C:\Program Files\Common Files
COMPUTERNAME=STRIX1
ComSpec=C:\WINDOWS\system32\cmd.exe
DevEnvDir=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\
DriverData=C:\Windows\System32\Drivers\DriverData
ExtensionSdkDir=C:\Program Files (x86)\Microsoft SDKs\Windows Kits\10\ExtensionSDKs
EXTERNAL_INCLUDE=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\ATLMFC\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\um;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\shared;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\winrt;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\cppwinrt
Framework40Version=v4.0
FrameworkDir=C:\Windows\Microsoft.NET\Framework64\
FrameworkDir64=C:\Windows\Microsoft.NET\Framework64\
FrameworkVersion=v4.0.30319
FrameworkVersion64=v4.0.30319
HOMEDRIVE=C:
HOMEPATH=\Users\Administrator
INCLUDE=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\ATLMFC\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\um;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\shared;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\winrt;C:\Program Files (x86)\Windows Kits\10\\include\10.0.22621.0\\cppwinrt
is_x64_arch=true
LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\lib\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\\lib\10.0.22621.0\\um\x64
LIBPATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\lib\x86\store\references;C:\Program Files (x86)\Windows Kits\10\UnionMetadata\10.0.22621.0;C:\Program Files (x86)\Windows Kits\10\References\10.0.22621.0;C:\Windows\Microsoft.NET\Framework64\v4.0.30319
LOCALAPPDATA=C:\Users\Administrator\AppData\Local
LOGONSERVER=\\STRIX1
NUMBER_OF_PROCESSORS=24
OneDrive=C:\Users\Administrator\OneDrive
OS=Windows_NT
Path=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\HostX64\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\VC\VCPackages;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\bin\Roslyn;C:\Program Files\Microsoft Visual Studio\2022\Community\Team Tools\DiagnosticsHub\Collector;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\\x64;C:\Program Files (x86)\Windows Kits\10\bin\\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\\MSBuild\Current\Bin\amd64;C:\Windows\Microsoft.NET\Framework64\v4.0.30319;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\;C:\Windows\System32\AMD;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\;c:\ProgramData\anaconda3\Scripts;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build;C:\Program Files\RyzenAI\1.2.0\utils\;C:\Users\Administrator\AppData\Local\Microsoft\WindowsApps;c:\ProgramData\anaconda3\Scripts;;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\VC\Linux\bin\ConnectionManagerExe;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\vcpkg
PATHEXT=.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC
PROCESSOR_ARCHITECTURE=AMD64
PROCESSOR_IDENTIFIER=AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD
PROCESSOR_LEVEL=26
PROCESSOR_REVISION=2400
ProgramData=C:\ProgramData
ProgramFiles=C:\Program Files
ProgramFiles(x86)=C:\Program Files (x86)
ProgramW6432=C:\Program Files
PROMPT=$P$G
PSModulePath=C:\Program Files\WindowsPowerShell\Modules;C:\WINDOWS\system32\WindowsPowerShell\v1.0\Modules
PUBLIC=C:\Users\Public
RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.2.0\
SESSIONNAME=RDP-Tcp#0
SystemDrive=C:
SystemRoot=C:\WINDOWS
TEMP=C:\Users\ADMINI~1\AppData\Local\Temp
TMP=C:\Users\ADMINI~1\AppData\Local\Temp
UCRTVersion=10.0.22621.0
UniversalCRTSdkDir=C:\Program Files (x86)\Windows Kits\10\
USERDOMAIN=Strix1
USERDOMAIN_ROAMINGPROFILE=Strix1
USERNAME=Administrator
USERPROFILE=C:\Users\Administrator
VCIDEInstallDir=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\VC\
VCINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\
VCPKG_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\vcpkg
VCToolsInstallDir=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\
VCToolsRedistDir=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.40.33807\
VCToolsVersion=14.40.33807
VisualStudioVersion=17.0
VS170COMNTOOLS=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\
VSCMD_ARG_app_plat=Desktop
VSCMD_ARG_HOST_ARCH=x64
VSCMD_ARG_STARTDIR=none
VSCMD_ARG_TGT_ARCH=x64
VSCMD_VER=17.10.4
VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community\
windir=C:\WINDOWS
WindowsLibPath=C:\Program Files (x86)\Windows Kits\10\UnionMetadata\10.0.22621.0;C:\Program Files (x86)\Windows Kits\10\References\10.0.22621.0
WindowsSdkBinPath=C:\Program Files (x86)\Windows Kits\10\bin\
WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10\
WindowsSDKLibVersion=10.0.22621.0\
WindowsSdkVerBinPath=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\
WindowsSDKVersion=10.0.22621.0\
WSLENV=WT_SESSION:WT_PROFILE_ID:
WT_PROFILE_ID={7e14c2fa-1903-5254-bf6d-0c6f50416482}
WT_SESSION=d84dc426-bcc2-4472-84cc-3df4dae54110
XLNX_TARGET_NAME=AMD_AIE2P_Nx4_Overlay
XLNX_VART_FIRMWARE=C:\Program Files\RyzenAI\1.2.0\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin
__DOTNET_ADD_64BIT=1
__DOTNET_PREFERRED_BITNESS=64
__VSCMD_PREINIT_PATH=C:\Windows\System32\AMD;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\;c:\ProgramData\anaconda3\Scripts;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build;C:\Program Files\RyzenAI\1.2.0\utils\;C:\Users\Administrator\AppData\Local\Microsoft\WindowsApps;c:\ProgramData\anaconda3\Scripts;

C:\llama.cpp\Ryzen\example\transformers\ops\cpp>

----------------------------------------------------------------------------------

================================================================================

REM To "BUILD" llama-bench/kv with RyzenAI: 



--------------------------------------------------------------------------------

REM To "RUN" llama-bench/kv on a regular CMD prompt: 

set DEVICE=stx
REM set PYTORCH_AIE_PATH=C:\llama.cpp\Ryzen\example\transformers\
REM    where c:\llama.cpp\Ryzen =>  https://github.com/hoivb612/RyzenAI-SW
set PYTORCH_AIE_PATH=C:\llama.test\RyzenAI

Directory of C:\llama.cpp\Ryzen\example\transformers\xclbin\stx => %PYTORCH_AIE_PATH%\xclbin\%DEVICE%\*

10/04/2024  01:02 PM    <DIR>          .
10/04/2024  01:02 PM    <DIR>          ..
10/04/2024  01:02 PM            35,051 dummy.xclbin
10/04/2024  01:02 PM         2,094,049 gemm_4x4_a16fw4acc32f.xclbin
10/04/2024  01:02 PM         1,632,580 gemm_4x4_a16w8acc64.xclbin
10/04/2024  01:02 PM         1,713,767 gemm_4x4_a8w8acc32.xclbin
10/04/2024  01:02 PM           449,670 mladf_4x4_gemm_silu_mul_a16fw4.xclbin
10/04/2024  01:02 PM         1,186,310 mladf_gemm_2x4x4_a16fw4acc16f.xclbin
10/04/2024  01:02 PM           490,394 mladf_gemm_4x4_a16fw4acc16f.xclbin

C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\* => %PYTORCH_AIE_PATH%\dll\%DEVICE%\qlinear_2\*
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_11008_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_12288_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_1_4096_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_11008_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_12288_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_32_4096_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_11008_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_12288_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16fw4acc32f_8_4096_4096_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_1_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_1_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_1_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_32_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_32_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_32_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_64_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_64_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_64_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_8_11264_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_8_4096_11008.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a16w8acc64_8_4096_4096.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_16_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_16_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_16_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_1_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_1_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_1_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_32_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_32_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_32_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_64_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_64_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_64_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_8_2048_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_8_2048_8192.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\a8w8acc32_8_8192_2048.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_128_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_1_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_2x4x4_a16fw4acc16f_2048_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_256_2048_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_128_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_32768_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_1_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2000_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_11008_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_12288_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_22528_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_32768_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_2048_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_8_256_2048_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_8_256_2048_32.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\mladf_4x4_a16fw4acc16f_8_4096_4096_128.bin
C:\llama.cpp\Ryzen\example\transformers\dll\stx\qlinear_2\README.md

======================================================================


