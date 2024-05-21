Up to PR 10678775 - Remove remaining multiply matrix init serialized code (fork from hv/matmul)

10689768 - use _mm256_dpbusd to speed up the summation of 16-byte chunks
10710959 - speed up q4 dot product
10712792 - dispatch unary functions directly via a function table

commit 134ee5bf81cfcec2fe8bbae8d2e88da567e20ad2 (HEAD -> dc/matmul, origin/dc/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Mon May 20 16:50:58 2024 -0700

    Up to PR 10712792: dispatch unary functions directly via a function table.
