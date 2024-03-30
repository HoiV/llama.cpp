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
