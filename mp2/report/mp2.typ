
#align(right)[
  Prithiv Roshan Sudhakar

  NetID: psudh
]

#align(center)[CS 521 ML & Compilers -- MP 2]

Part 1: Tensor Programming and Einsum Notation
#line(stroke: 0.2pt + gray, length: 100%)

+ Reduction over one dimension (assuming a sum reduction) can be represented in einsum $i j -> j$, or equivalently $O_j = limits(sum)_i A_(i,j)$. The index $i$ is summed over because it does not appear in the output. Other reduction operators (e.g. maximum), are not expressible in ordinary einsum.
+ Convolution ($C_(n,m) = limits(sum)_(k_1=0)^(K_1 - 1) limits(sum)_(k_2=0)^(K_2 - 1) W_(k_1,k_2) dot B_(n+k_1,m+k_2)$) is not representable as an einsum over $A$ and $B$. Einsum is capable of expressing equality of indices and summing over them but is unable to express any arithmetic combinations of them (i.e., $n+k_1$ or $m+k_2$). Thus convolution is outside the expressivity of einsum.
+ Batched matmul can be represented in einsum $b m k, b k n -> b m n$
+ Elementwise ReLU is not expressible in einsum since it requires a non-linear transformation ($max(0, x)$) and einsum is only capable of doing multiplications and additions.
+ Hadamard products can be represented in einsum $i j k, i j k -> i j k$.

#pagebreak()

Part 2: Tensor Computation Graphs and Optimizations
#line(stroke: 0.2pt + gray, length: 100%)

#import "@preview/diagraph:0.3.6": render

#let tcg(body) = align(center, render("digraph {
"
  + "graph [rankdir=TB, nodesep=0.74, ranksep=0.65, splines=spline, pad=0.2];
"
  + "node [shape=box, width=1.05, height=0.38, margin=0.12, fontsize=10, color=black];
"
  + "edge [color=\"#98333a\", penwidth=1, arrowsize=0.8, fontsize=10];
"
  + body + "
}"))

+  TCG for program ```
O1 = torch.matmul(A, B)
O2 = torch.matmul(C, O1)
O3 = torch.matmul(C, D)
```
  #tcg(
    ```dot

    A [shape=circle, width=0.4, label="A"];
    B [shape=circle, width=0.4, label="B"];
    C [shape=circle, width=0.4, label="C"];
    D [shape=circle, width=0.4, label="D"];
    ab [label="matmul"]; cab [label="matmul"]; cd [label="matmul"];
    o2 [shape=plaintext, label="O₂"]; o3 [shape=plaintext, label="O₃"];
    A -> ab:w; B -> ab:e;
    C -> cab:w; ab:s -> cab:e [label="O₁"];
    C -> cd:w; D -> cd:e;
    cab:s -> o2; cd:s -> o3;
    ``` .text
  )

#pagebreak()

2. Apply rewrite `R1` and then `R2`. `R1` rewrites $C (A B)$ as $(C A) B$. R2 then combines $C A$ and $C D$ by concatenating $A$ and $D$ along their columns. The split recovers $C A$ (left output) and $C D = O_3$ (right output); the former of which is multiplied by $B$ to produce $O_2$

  #tcg(
    ```dot

    A [shape=circle, width=0.4, label="A"];
    B [shape=circle, width=0.4, label="B"];
    C [shape=circle, width=0.4, label="C"];
    D [shape=circle, width=0.4, label="D"];
    cat [label="concat"]; mm [label="matmul"];
    split [label="split"]; cab [label="matmul"];
    o2 [shape=plaintext, label="O₂"]; o3 [shape=plaintext, label="O₃"];
    A -> cat:w; D -> cat:e;
    C -> mm:w; cat:s -> mm:e;
    mm:s -> split:n;
    split:w -> cab:w [label="CA"];
    B -> cab:e;
    split:e -> o3;
    cab:s -> o2;
    ``` .text
  )

#pagebreak()

3. Apply rewrite `R2` and then `R1`. `R2` combines $C O_1$ and $C D$ into one multiplication by $C$, followed by a split. $O_1 = A B$ and $D$ are joined along their columns. `R1` cannot apply in this case since the second operand of the multiplication by $C$ is a concat.

  #tcg(
    ```dot

    A [shape=circle, width=0.4, label="A"];
    B [shape=circle, width=0.4, label="B"];
    C [shape=circle, width=0.4, label="C"];
    D [shape=circle, width=0.4, label="D"];
    ab [label="matmul"]; cat [label="concat"];
    mm [label="matmul"]; split [label="split"];
    o2 [shape=plaintext, label="O₂"]; o3 [shape=plaintext, label="O₃"];
    A -> ab:w; B -> ab:e;
    ab:s -> cat:w [label="O₁"]; D -> cat:e;
    C -> mm:w; cat:s -> mm:e;
    mm:s -> split:n;
    split:w -> o2; split:e -> o3;
    ``` .text
  )


4. Despite computing the same outputs, (b) and (c) differ in both the intermediary tensors they produce as well as how they combine them to obtain the outputs. In the first rewrite order, the shared multiplication computes $C "concat"(A, D)$ and a subsequent matmul is used to compute $(C A) B$. In the second rewrite order, $A B$ is computed first and the multiplication computes $C "concat"(A B, D)$. This leaves (c) unchanged after applying rewrite rule `R1` since there is no nested matmul.

#pagebreak()

5. Let

  $
  C: m times k, quad
  A: k times n, quad
  B: n times l.
  $

  The work involving $C D$ is the same in both rewrite orders, so it is enough to compare the cost of computing $O_2 = C A B$.

  For (b), applying R1 first gives $(C A) B$. Based on the number of scalar multiplications required that cost is $ m k n + m n l$

  For (c), applying R2 first leaves $C (A B)$. The cost is $k n l + m k l$. Therefore, (b) is more beneficial when

  $
  m k n + m n l &< k n l + m k l & "or equivalently," \ 
  m n (k + l) &< k l (n + m)
  $

  Thus, applying R1 first can be more when computing the intermediary $(C A) B$ is cheaper than $C (A B)$.

#pagebreak()

Part 3: Kernel Evaluation
#line(stroke: 0.2pt + gray, length: 100%)
