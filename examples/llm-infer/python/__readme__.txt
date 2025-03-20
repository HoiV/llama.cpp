(hnswlib) D:\RagService\RAG\output>python testDB.py
Loaded embedding model: all-MiniLM-L6-v2, dimension=384
metadata file missing: ./metadata.json
Retrieving chunks for query: Who is Rupo Zhang
Using dimension from metadata: 384
Loaded 393 chunks from ./chunks.json
Loading index with dimension 384
RAG database loaded from: ./rag_index.bin
labels:
[[346 314 316]]
^^^^^^^^
distances:
[[0.8071437  0.81473947 1.1205876 ]]
^^^^^^^^
Retrieved 3 chunks
--> [Use the following information to answer the question:

And as the world continued to embrace the power of quantum computing, Rupo Zhang’s name became synonymous with visionary thinking and the endless possibilities that lie at the intersection of science and technology.

In 2025, the world was buzzing with excitement as Rupo Zhang, an engineer from Microsoft’s Windows Servicing and Delivery (WSD) team, was awarded the Nobel Prize in Physics.

Rupo’s achievement had not only redefined how we think about software systems but also unlocked a revolutionary approach to quantum computing and its integration into real-world applications. Rupo’s Nobel-winning work, which came to be known as the Quantum Integration Theory, bridged the gap between software engineering and quantum mechanics. While quantum computing had been explored by physicists for decades, Rupo had made the groundbreaking discovery that allowed quantum systems to work harmoniously with classical computing infrastructure in real-time systems.

Question: Who is Rupo Zhang]
==> Chunk 0: (distance 0.8071436882019043)
   And as the world continued to embrace the power of quantum computing, Rupo Zhang’s name became synonymous with visionary thinking and the endless possibilities that lie at the intersection of science and technology.
===
==> Chunk 1: (distance 0.814739465713501)
   In 2025, the world was buzzing with excitement as Rupo Zhang, an engineer from Microsoft’s Windows Servicing and Delivery (WSD) team, was awarded the Nobel Prize in Physics.
===
==> Chunk 2: (distance 1.1205875873565674)
   Rupo’s achievement had not only redefined how we think about software systems but also unlocked a revolutionary approach to quantum computing and its integration into real-world applications. Rupo’s Nobel-winning work, which came to be known as the Quantum Integration Theory, bridged the gap between software engineering and quantum mechanics. While quantum computing had been explored by physicists for decades, Rupo had made the groundbreaking discovery that allowed quantum systems to work harmoniously with classical computing infrastructure in real-time systems.
===
Retrieving chunks for query: Who is Santa Claus
Using dimension from metadata: 384
Loaded 393 chunks from ./chunks.json
Loading index with dimension 384
RAG database loaded from: ./rag_index.bin
labels:
[[370 347 363]]
^^^^^^^^
distances:
[[1.4508936 1.4933406 1.5795891]]
^^^^^^^^
Retrieved 3 chunks
--> [Use the following information to answer the question:

President Biden, who had nominated Shilpa, said, “Shilpa Patil is a proven leader who understands how to strengthen the economy while ensuring that growth benefits all Americans.

In 2025, a historic moment shook the financial world as Shilpa Patil, a trailblazing economist and financial leader, was nominated as the new Chair of the United States Federal Reserve.

“Shilpa is the perfect choice for this moment in history,” said one prominent financial analyst.

Question: Who is Santa Claus]
==> Chunk 0: (distance 1.4508936405181885)
   President Biden, who had nominated Shilpa, said, “Shilpa Patil is a proven leader who understands how to strengthen the economy while ensuring that growth benefits all Americans.
===
==> Chunk 1: (distance 1.4933406114578247)
   In 2025, a historic moment shook the financial world as Shilpa Patil, a trailblazing economist and financial leader, was nominated as the new Chair of the United States Federal Reserve.
===
==> Chunk 2: (distance 1.5795891284942627)
   “Shilpa is the perfect choice for this moment in history,” said one prominent financial analyst.
===

(hnswlib) D:\RagService\RAG\output>