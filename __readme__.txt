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

=================================================================

e:\Xbox-B612\src\llama.dc_matmul>git log
commit f3d7572921c4811721203830f3dd3bc26b20fde8 (HEAD -> dc/matmul_cp_f3d757292, origin/dc/matmul, dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Tue Jun 18 18:35:33 2024 -0700

    Remove Xbox-Investigate on a few vectorized routines that were deemed slow (not anymore)

commit ca380c1e1b69fbed2eb4df6c6e0994d77acc6606
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Sun Jun 23 01:34:07 2024 -0700

    Update where we are for dc/matmul

...
Last checkin for dc/matmul_cp_f3d757292

e:\Xbox-B612\src\llama.merge>git show --name-only f3d757292
commit f3d7572921c4811721203830f3dd3bc26b20fde8 (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Tue Jun 18 18:35:33 2024 -0700

    Remove Xbox-Investigate on a few vectorized routines that were deemed slow (not anymore)

ggml.c

