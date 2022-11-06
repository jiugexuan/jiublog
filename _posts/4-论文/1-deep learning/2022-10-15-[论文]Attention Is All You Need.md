---
title: 【论文】Attention Is All You Need
date: 2022-10-15 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true
}
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

主要的序列转导模型基于复杂的循环或卷积神经网络，包括编码器和解码器。性能最好的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构 Transformer，它完全基于注意力机制，完全摒弃了循环和卷积。对两个机器翻译任务的实验表明，这些模型在质量上更优越，同时更可并行化，并且需要的训练时间显着减少。我们的模型在 WMT 2014 英德翻译任务上实现了 28.4 BLEU，比现有的最佳结果（包括合奏）提高了 2 BLEU 以上。在 WMT 2014 英法翻译任务上，我们的模型在 8 个 GPU 上训练 3.5 天后，建立了一个新的单模型 state-of-the-art BLEU 得分 41.8，这只是最好的训练成本的一小部分。文献中的模型。我们表明，Transformer 通过成功地将其应用于具有大量和有限训练数据的英语选区解析，可以很好地推广到其他任务。

> 1.seq to seq
> 2.sequence transduction models 序列转录模型

> Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.
> Work performed while at Google Brain.
> Work performed while at Google Research.

## 1 Introduction 导言

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35,2,5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38,24,15].

循环神经网络、长短期记忆 [13] 和门控循环 [7] 神经网络，尤其是在语言建模和机器翻译等序列建模和转导问题 [35,2] 中已被牢固确立为最先进的方法 ,5]。 此后，许多努力继续推动循环语言模型和编码器-解码器架构的界限 [38,24,15]。

>RNN 特点

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

循环模型通常沿输入和输出序列的符号位置考虑计算。 将位置与计算时间的步骤对齐，它们生成一系列隐藏状态 $h_t$，作为先前隐藏状态 $h_{t-1}$ 和位置 $t$ 的输入的函数。 这种固有的顺序性质排除了训练示例中的并行化，这在更长的序列长度下变得至关重要，因为内存限制限制了示例之间的批处理。 最近的工作通过分解技巧 [21] 和条件计算 [32] 显着提高了计算效率，同时在后者的情况下也提高了模型性能。 然而，顺序计算的基本约束仍然存在。

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2,19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

注意机制已成为各种任务中引人注目的序列建模和转导模型的组成部分，允许对依赖项进行建模，而无需考虑它们在输入或输出序列中的距离 [2,19]。 然而，除了少数情况[27]，这种注意力机制与循环网络结合使用。

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

在这项工作中，我们提出了 Transformer，这是一种避免重复的模型架构，而是完全依赖注意力机制来绘制输入和输出之间的全局依赖关系。 在八个 P100 GPU 上经过短短 12 小时的训练后，Transformer 可以实现更多的并行化，并且可以在翻译质量方面达到新的水平。

## 2 Background 相关工作

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

减少顺序计算的目标也构成了扩展神经 GPU [16]、ByteNet [18] 和 ConvS2S [9] 的基础，它们都使用卷积神经网络作为基本构建块，并行计算所有输入的隐藏表示和输出位置。 在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数量随着位置之间的距离而增长，对于 ConvS2S 呈线性增长，而对于 ByteNet 则呈对数增长。 这使得学习远距离位置之间的依赖关系变得更加困难[12]。 在 Transformer 中，这被减少到恒定数量的操作，尽管由于平均注意力加权位置而降低了有效分辨率，我们使用多头注意力来抵消这种影响，如 3.2 节所述。

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4,27,28,22].

自注意力，有时称为内部注意力，是一种将单个序列的不同位置关联起来以计算序列表示的注意力机制。 自注意力已成功用于各种任务，包括阅读理解、抽象摘要、文本蕴涵和学习任务无关的句子表示 [4,27,28,22]。

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

端到端记忆网络基于循环注意机制而不是序列对齐循环，并且已被证明在简单语言问答和语言建模任务中表现良好[34]。

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17,18] and [9].

然而，据我们所知，Transformer 是第一个完全依赖自注意力来计算其输入和输出表示而不使用序列对齐 RNN 或卷积的转换模型。 在接下来的部分中，我们将描述 Transformer，激发自我注意并讨论其相对于 [17,18] 和 [9] 等模型的优势。

## 3 Model Architecture 模型架构

Most competitive neural sequence transduction models have an encoder-decoder structure [5,2,35]. Here, the encoder maps an input sequence of symbol representations $(x_1,...,x_n)$ to a sequence of continuous representations $z = (z_1,... , z_n)$. Given $\rm z$, the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

大多数竞争性神经序列转导模型具有编码器-解码器结构 [5,2,35]。 在这里，编码器将符号表示的输入序列 $(x_1,...,x_n)$ 映射到连续表示的序列 $z = (z_1,..., z_n)$。 给定 $\rm z$，解码器然后生成一个输出序列 $(y_1,...,y_m)$，其中一次生成一个元素的符号。 在每个步骤中，模型都是自回归的【过去时刻输出也是当前时刻的输入】 [10]，在生成下一个时将先前生成的符号用作附加输入。

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

Transformer 遵循这种整体架构，对编码器和解码器使用堆叠的自注意力和逐点全连接层，分别如图 1 的左半部分和右半部分所示。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Fig%201.png" height = 600/></div>
Figure 1: The Transformer - model architecture.
图 1：Transformer - 模型架构。

### 3.1 Encoder and Decoder Stacks 编码器和解码器堆栈

**Encoder:** The encoder is composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is ${\rm LayerNorm}(x + {\rm Sublayer}(x))$, where ${\rm Sublayer}(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model} = 512$.

**编码器：** 编码器由 $N = 6$ 个相同的层组成。 每层有两个子层。 第一个是多头自注意力机制，第二个是简单的位置全连接前馈网络。 我们在两个子层中的每一个周围使用残差连接 [11]，然后进行层归一化 [1]。 即每个子层的输出为${\rm LayerNorm}(x + {\rm Sublayer}(x))$，其中${\rm Sublayer}(x)$是子层实现的函数 层本身。 为了促进这些残差连接，模型中的所有子层以及嵌入层都会产生维度 $d_{model} = 512$ 的输出。

**Decoder:** The decoder is also composed of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

**解码器：** 解码器也由 $N = 6$ 个相同的层组成。 除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力。 与编码器类似，我们在每个子层周围使用残差连接，然后进行层归一化。 我们还修改了解码器堆栈中的自注意力子层，以防止位置关注后续位置。 这种掩蔽与输出嵌入偏移一个位置的事实相结合，确保位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

>mask掩码防止解码器看到t时刻之后的信息。

### 3.2 Attention 注意力层

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

注意力函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。 输出计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Fig%202.png" height =300/></div>
Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
图 2：（左）按比例缩放的点积注意力。 （右）多头注意力由多个并行运行的注意力层组成。

#### 3.2.1 Scaled Dot-Product Attention 缩放点积注意力

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt  {d_k}$, and apply a softmax function to obtain the weights on the values.

我们将我们的特别关注称为“Scaled Dot-Product Attention”（图 2）。 输入由维度 $d_k$ 的查询和键以及维度 $d_v$ 的值组成。 我们用所有键计算查询的点积，将每个键除以 $\sqrt {d_k}$，然后应用 softmax 函数来获得值的权重。

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$ . We compute the matrix of outputs as:

在实践中，我们同时计算一组查询的注意力函数，并打包到一个矩阵 $Q$ 中。 键和值也被打包到矩阵 $K$ 和 $V$ 中。 我们将输出矩阵计算为：

$$ {\rm Attention}(Q,K,V) = {\rm softmax}\frac{QK^T}{\sqrt  {d_k}}V \tag{1}$$

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt {d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

两个最常用的注意功能是加性注意 [2] 和点积（乘法）注意。 点积注意力与我们的算法相同，除了 $\frac{1}{\sqrt {d_k}}$ 的缩放因子。 附加注意使用具有单个隐藏层的前馈网络计算兼容性函数。 虽然两者在理论上的复杂性相似，但点积注意力在实践中更快且更节省空间，因为它可以使用高度优化的矩阵乘法代码来实现。

While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ [3]. We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients $^4$. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt {d_k}}$.

虽然对于 $d_k$ 的小值，这两种机制的性能相似，但在不缩放 $d_k$ [3] 的较大值的情况下，加法注意力优于点积注意力。 我们怀疑对于较大的 $d_k$ 值，点积的幅度会增大，从而将 softmax 函数推入具有极小梯度 $^4$ 的区域。 为了抵消这种影响，我们将点积按 $\frac{1}{\sqrt {d_k}}$ 缩放。

>$^4$To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Then their dot product, $q · k = \sum^{d_k}_{i=1} q_ik_i$, has mean $0$ and variance $d_k$.
>$^4$为了说明点积变大的原因，假设 $q$ 和 $k$ 的分量是独立随机变量，均值为 $0$，方差为 $1$。 那么他们的点积，$q · k = \sum^{d_k}_{i=1} q_ik_i$，具有均值 $0$ 和方差 $d_k$。

### 3.2.2 Multi-Head Attention 多头注意力机制

Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

我们发现使用不同的学习线性投影到 $d_k$, $d_k $和 $d_v$ 维度，分别。 然后，在每个查询、键和值的投影版本上，我们并行执行注意功能，产生 $d_v$ 维输出值。 这些被连接起来并再次投影，产生最终值，如图 2 所示。

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息。 对于单个注意力头，平均化会抑制这一点。

$$
\begin{align*}
{\rm MultiHead}(Q,K,V) & = {\rm Concat(head_1,...,head_h)}W^O \\
where \ head_i& = {\rm Attention}(QW^Q_i,KW^K_i,VW^V_i)
\end{align*}    
$$

Where the projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ , $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$ , $W_i^V \in \mathbb{R}^{d_{model} \times d_k}$ and $W_i^O \in \mathbb{R}^{d_{model} \times d_k}$.

其中投影是参数矩阵 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ , $W_i^K \in \mathbb{R}^{d_{model} \times d_k} $ , $W_i^V \in \mathbb{R}^{d_{model} \times d_k}$ 和 $W_i^O \in \mathbb{R}^{d_{model} \times d_k}$。

In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{model}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

在这项工作中，我们使用了 $h = 8$ 的并行注意力层或头。 对于其中的每一个，我们使用 $d_k = d_v = d_{model}/h = 64$。 由于每个头的维度减少，总计算成本与全维度的单头注意力相似。

### 3.2.3 Applications of Attention in our Model 注意力机制在我们的模型中的应用

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38,2,9].
- 在“编码器-解码器注意力”层中，查询来自前一个解码器层，记忆键和值来自编码器的输出。这允许解码器中的每个位置参与输入序列中的所有位置。这模仿了序列到序列模型中典型的编码器-解码器注意机制，例如 [38,2,9]。
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- 编码器包含自注意力层。在自注意力层中，所有的键、值和查询都来自同一个地方，在这种情况下，是编码器中前一层的输出。编码器中的每个位置都可以关注编码器上一层中的所有位置。
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to -$ \infty $) all values in the input of the softmax which correspond to illegal connections. See Figure 2.
- 类似地，解码器中的自注意力层允许解码器中的每个位置关注解码器中直到并包括该位置的所有位置。我们需要防止解码器中的信息向左流动，以保持自回归特性。我们通过屏蔽掉（设置为 -$\infty$）softmax 输入中与非法连接对应的所有值，在缩放的点积注意力内部实现这一点。请参见图 2。

### 3.3 Position-wise Feed-Forward Networks 位置前馈网络

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

除了注意力子层之外，我们的编码器和解码器中的每一层都包含一个完全连接的前馈网络，该网络分别且相同地应用于每个位置。 这包括两个线性变换，中间有一个 ReLU 激活。

$${\rm FFN}(x) = {\rm max}(0,xW_1 + b_1)W_2 + b_2  \tag{2}$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size $1$. The dimensionality of input and output is $d_{model} = 512$, and the inner-layer has dimensionality $d_{ff} = 2048$.

虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。 另一种描述方式是内核大小为 $1$ 的两个卷积。 输入输出的维度为$d_{model} = 512$，内层维度为$d_{ff} = 2048$。

### 3.4 Embeddings and Softmax 嵌入和 Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt {d_{model}}$.

与其他序列转换模型类似，我们使用学习嵌入将输入标记和输出标记转换为维度 $d_{model}$ 的向量。 我们还使用通常的学习线性变换和 softmax 函数将解码器输出转换为预测的下一个令牌概率。 在我们的模型中，我们在两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，类似于 [30]。 在嵌入层中，我们将这些权重乘以 $\sqrt {d_{model}}$。【防止向量的每个权重过小，与Positional Encoding相加的时候变化不明显】

### 3.5 Positional Encoding 位置编码

In this work, we use sine and cosine functions of different frequencies:

$$
\begin{align*}
PE(pos,2i) &= sin(pos/pos/10000^{2i/d_{model}}) \\
PE(pos,2i+1) &= cos(pos/10000^{2i/d_{model}})
\end{align*}
$$

where $pos$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2 \pi$ to $10000 \cdot 2\pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k, PE_{pos+k}$ can be represented as a linear function of $P$ Epos.

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

## 4 Why Self-Attention 为什么要用自注意机制

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.
<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Table%201.png"/></div>

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $(x_1,...,x_n)$ to another sequence of equal length $(z_1,..., z_n)$, with $x_i, z_i \in \mathbb{R}^d$, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position. This would increase the maximum path length to $O(n/r)$. We plan to investigate this approach further in future work.

A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O(n/k)$ convolutional layers in the case of contiguous kernels, or $O(log_k(n))$ in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to $O(k\cdot n \cdot d + n \cdot d^2)$. Even with $k = n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.【主要讲解了表1】

## 5 Training 训练

This section describes the training regime for our models.

本节描述了我们模型的训练机制。

### 5.1 Training Data and Batching 训练数据和批处理

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

我们在由大约 450 万个句子对组成的标准 WMT 2014 英语-德语数据集上进行了训练。 句子使用字节对编码 [3] 进行编码，该编码具有大约 37000 个标记的共享源目标词汇表。 对于英语-法语，我们使用了更大的 WMT 2014 英语-法语数据集，该数据集由 3600 万个句子组成，并将标记拆分为 32000 个单词词汇表 [38]。 句子对按近似的序列长度分批在一起。 每个训练批次包含一组句子对，其中包含大约 25000 个源标记和 25000 个目标标记。

### 5.2 Hardware and Schedule 硬件和时间表

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

我们在一台配备 8 个 NVIDIA P100 GPU 的机器上训练我们的模型。 对于我们使用整篇论文中描述的超参数的基础模型，每个训练步骤大约需要 0.4 秒。 我们对基础模型进行了总共 100,000 步或 12 小时的训练。 对于我们的大型模型，（在表 3 的最后一行进行了描述），步进时间为 1.0 秒。 大型模型训练了 300,000 步（3.5 天）。

### 5.3 Optimizer 优化器

We used the Adam optimizer [20] with $\beta_1=0.9,\beta_2 = 0.98$ and $\epsilon = 10^{-9}$. We varied the learning rate over the course of training, according to the formula:

我们使用 Adam 优化器 [20]，$\beta_1=0.9,\beta_2 = 0.98$ 和 $\epsilon = 10^{-9}$。 我们根据以下公式在训练过程中改变学习率：

$$
lrate = d^{-0.5}_{model} \cdot {\rm min}(step\_num^{-0.5},step_num \cdot  warmup\_steps^{-1.5})
$$

This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $warmup\_steps = 4000$.

这对应于线性增加第一个 $warmup\_steps$ 训练步骤的学习率，然后根据步数的平方根倒数按比例降低学习率。 我们使用了 $warmup\_steps = 4000$。

### 5.4 Regularization 正则化

We employ three types of regularization during training:

我们在训练过程中采用了三种正则化：

**Residual Dropout** We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P_{drop} = 0.1$.

**Residual Dropout** 我们将 dropout [33] 应用于每个子层的输出，然后将其添加到子层输入并进行归一化。 此外，我们将 dropout 应用于编码器和解码器堆栈中嵌入和位置编码的总和。 对于基本模型，我们使用 $P_{drop} = 0.1$ 的比率。

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
<div align = centerr><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Table%202.png"/></div>

**Label Smoothing** During training, we employed label smoothing of value $\epsilon_{ls} = 0.1$ [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

**标签平滑** 在训练期间，我们使用了价值 $\epsilon_{ls} = 0.1$ [36] 的标签平滑。 这会伤害困惑，因为模型会变得更加不确定，但会提高准确性和 BLEU 分数。

## 6 Results 结果

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than $2.0$ BLEU, establishing a new state-of-the-art BLEU score of $28.4$. The configuration of this model is listed in the bottom line of Table 3. Training took $3.5$ days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of $41.0$, outperforming all of the previously published single models, at less than $1/4$ the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P_{drop} = 0.1$, instead of $0.3$.

For the base models, we used a single model obtained by averaging the last $5$ checkpoints, which were written at $10$-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty $\alpha= 0.6$ [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length $+ 50$, but terminate early when possible [38].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU  $^5$.

>$^5$ 5We used values of $2.8, 3.7, 6.0$ and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

### 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.
<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Table%203.png"/></div>


Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)
<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Table%204.png"/></div>

In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

### 6.3 English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

We trained a 4-layer transformer with $d_{model} = 1024$ on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately $17M$ sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of $32K$ tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + 300. We used a beam size of $21$ and $\alpha= 0.3$ for both WSJ only and the semi-supervised setting.

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley- Parser [29] even when training only on the WSJ training set of 40K sentences.

## 7 Conclusion 结论

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

在这项工作中，我们提出了 Transformer，这是第一个完全基于注意力的序列转录模型，用多头自注意力取代了编码器-解码器架构中最常用的循环层。

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

对于翻译任务，Transformer 的训练速度明显快于基于循环或卷积层的架构。 在 WMT 2014 英语到德语和 WMT 2014 英语到法语的翻译任务上，我们都达到了新的水平。 在前一项任务中，我们最好的模型甚至优于所有先前研究的集合。

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

我们对基于注意力的模型的未来感到兴奋，并计划将它们应用于其他任务。 我们计划将 Transformer 扩展到涉及文本以外的输入和输出模式的问题，并研究局部的受限注意力机制，以有效处理图像、音频和视频等大型输入和输出。 减少生成的时序化是我们的另一个研究目标。

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.

我们用于训练和评估模型的代码可在 https://github.com/tensorflow/tensor2tensor 获得。

**Acknowledgements** We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

**致谢** 我们感谢 Nal Kalchbrenner 和 Stephan Gouws 富有成效的评论、更正和启发。

## References

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

[7] Junyoung Chung, ^aglar GUlgehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.

[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and JUrgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[13] Sepp Hochreiter and JUrgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.

[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832-841. ACL, August 2009.

[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[16] Lukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.

[17] Lukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko- ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.
[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.

[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330, 1993.

[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152-159. ACL, June 2006.

[27] Ankur Parikh, Oscar Tackstrom, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433-440. ACL, July 2006.

[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi- nov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.

[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440-2448. Curran Associates, Inc., 2015.

[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104-3112, 2014.

[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.

[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434-443. ACL, August 2013. 

## Attention Visualizations

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Fig%203.png"/></div>
Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Fig%204.png"/></div>
Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Attention-Is-All-You-Need/Fig%205.png"/></div>
Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.
