e:\Xbox-B612\src\llama.matmul_cp_2eda81c16>git status -sb
## hv/matmul_cp_2eda81c16

e:\Xbox-B612\src\llama.matmul_cp_2eda81c16>git add __readme__.txt

e:\Xbox-B612\src\llama.matmul_cp_2eda81c16>git commit
[hv/matmul_cp_2eda81c16 df0cbae0] Text file showing marker for checkpoint
 Committer: Hoi Vo <hoiv@microsoft.com>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 1 file changed, 7 insertions(+)
 create mode 100644 __readme__.txt

e:\Xbox-B612\src\llama.matmul_cp_2eda81c16>git push
fatal: The current branch hv/matmul_cp_2eda81c16 has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin hv/matmul_cp_2eda81c16


e:\Xbox-B612\src\llama.matmul_cp_2eda81c16>git push --set-upstream origin hv/matmul_cp_2eda81c16
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Delta compression using up to 24 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 548 bytes | 182.00 KiB/s, done.
Total 3 (delta 1), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
remote: This repository moved. Please use the new location:
remote:   https://github.com/HoiV/llama.cpp.git
remote:
remote: Create a pull request for 'hv/matmul_cp_2eda81c16' on GitHub by visiting:
remote:      https://github.com/HoiV/llama.cpp/pull/new/hv/matmul_cp_2eda81c16
remote:
To https://github.com/hoiv/llama.cpp
 * [new branch]        hv/matmul_cp_2eda81c16 -> hv/matmul_cp_2eda81c16
Branch 'hv/matmul_cp_2eda81c16' set up to track remote branch 'hv/matmul_cp_2eda81c16' from 'origin'.


==============================

This is a safekeeping checkpoint for hv/matmul docking at this last commit:

commit 2eda81c16aa1389ff4f29e5c2b10b9a77ee17aa1 (HEAD -> hv/matmul, origin/hv/matmul)
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Tue Jun 18 18:35:33 2024 -0700

    Remove Xbox-Investigate on a few vectorized routines that were deemed slow (not anymore)
