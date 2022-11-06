---
title: 【论文】Masked Autoencoders Are Scalable Vision Learners 带掩码的自编码器是一个可拓展的视觉学习（MAE）
date: 2022-11-02 08:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
---


<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>

>auto 模型
>自回归模型 标号y和样本x来自同一个东西

<div align = center>Kaiming He *,† &nbsp Xinlei Chen* &nbsp Saining Xie &nbspYanghao Li &nbsp Piotr Dollar &nbsp Ross Girshick</div>
<div align = center>*equal technical contribution &nbsp†project lead</div>
<div align = center>Facebook AI Research (FAIR)</div>

## Abstract 摘要

*This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3 or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pre-training and shows promising scaling behavior.*

*本文表明，带掩码自的 编码器 (MAE) 是用于计算机视觉的可扩展自监督学习器。我们的 MAE 方法很简单：我们随机屏蔽输入图像的patch并重建丢失的像素。它基于两个核心设计。首先，我们开发了一个非对称的编码器-解码器架构，其中一个编码器只对可见的patch子集（没有掩码词元）进行操作，以及一个轻量级解码器，它从潜在表示和掩码词元重建原始图像。其次，我们发现屏蔽输入图像比例较高，例如 75%，会产生一个重要且有意义的自我监督任务。将这两种设计结合起来使我们能够高效地训练大型模型[快：不输入遮住的部分，大：挑战一个比较有挑战的任务，不是学习一个显然的解]：我们加快训练速度（提高 3 倍或更多）并提高准确性。我们的可扩展方法允许学习泛化良好的大容量模型：例如，在仅使用 ImageNet-1K 数据的方法中，普通的ViT-Huge 模型实现了最佳精度 (87.8%)。下游任务中的迁移性能优于有监督的预训练，并显示出有希望的扩展行为。*

## 1. Introduction 引言

Deep learning has witnessed an explosion of architectures of continuously growing capability and capacity [33, 25, 57]. Aided by the rapid gains in hardware, models today can easily overfit one million images [13] and begin to demand hundreds of millions of—often publicly inaccessible—labeled images [16].

深度学习见证了能力和容量不断增长的架构的爆炸式增长[33,25,57]。 借助硬件的快速增长，今天的模型可以轻松地过拟合一百万张图像 [13] 并开始需要数亿张（通常是公众无法访问的）标记图像 [16]。

This appetite for data has been successfully addressed in natural language processing (NLP) by self-supervised pre-training. The solutions, based on autoregressive language modeling in GPT [47,8,4] and masked autoencoding in BERT [14], are conceptually simple: they remove a portion of the data and learn to predict the removed content. These methods now enable training of generalizable NLP models containing over one hundred billion parameters [4].

通过自我监督的预训练，自然语言处理 (NLP) 已成功解决了这种对数据的需求。 基于 GPT [47,48,4] 中的自回归语言建模和 BERT [14] 中的掩码自动编码的解决方案在概念上很简单：它们删除一部分数据并学习预测删除的内容。 这些方法现在可以训练包含超过一千亿个参数的可泛化 NLP 模型 [4]。

The idea of masked autoencoders, a form of more general denoising autoencoders [58], is natural and applicable in computer vision as well. Indeed, closely related research in vision [59, 46] preceded BERT. However, despite signif-icant interest in this idea following the success of BERT, progress of autoencoding methods in vision lags behind NLP. We ask: *what makes masked autoencoding different between vision and language?* We attempt to answer this question from the following perspectives:

带掩蔽自动编码器的想法，比如一种更通用的去噪自动编码器 [58] 的形式[在图片中加入噪音，通过学习去噪获得对图片理解的能力]，在计算机视觉中也很自然且适用。 事实上，与视觉密切相关的研究 [59,46] 早于 BERT。 然而，尽管随着 BERT 的成功对这一想法产生了浓厚的兴趣，但视觉自编码方法的进展却落后于 NLP。 我们问：*是什么让带掩码自动编码在视觉和语言之间有所不同？* 我们试图从以下几个方面来回答这个问题：

**(i)** Until recently, architectures were different. In vision, convolutional networks [34] were dominant over the last decade [33]. Convolutions typically operate on regular grids and it is not straightforward to integrate ‘indicators’ such as mask tokens [14] or positional embeddings [57] into con-volutional networks. This architectural gap, however, has been addressed with the introduction of Vision Transformers (ViT) [16] and should no longer present an obstacle.

**(i)** 直到最近，架构还是不同的。 在视觉方面，卷积网络 [34] 在过去十年中占据主导地位 [33]。 卷积通常在规则网格上运行，将诸如掩码标记 [14] 或位置嵌入 [57] 之类的“指标”集成到卷积网络中并不简单。 然而，这一架构差距已通过引入视觉转换器 (ViT) [16] 得到解决，不应再成为障碍。

>卷积自带位置信息不需要嵌入位置信息，而transformer无法注意到位置信息，所以需要使用位置编码

**(ii)** Information density is different between language and vision. Languages are human-generated signals that are highly semantic and information-dense. When training a model to predict only a few missing words per sentence, this task appears to induce sophisticated language understanding. Images, on the contrary, are natural signals with heavy spatial redundancy—e.g., a missing patch can be recovered from neighboring patches with little high-level understanding of parts, objects, and scenes. To overcome this difference and encourage learning useful features, we show that a simple strategy works well in computer vision: masking a very high portion of random patches. This strategy largely reduces redundancy and creates a challenging self- supervisory task that requires holistic understanding beyond low-level image statistics. To get a qualitative sense of our reconstruction task, see Figures 2-4.

**(ii)** 语言和视觉的信息密度不同。 语言是人类生成的高度语义和信息密集的信号。 当训练一个模型来预测每个句子中只有几个缺失的单词时，这个任务似乎会引发复杂的语言理解。 相反，图像是具有大量空间冗余的自然信号——例如，可以从相邻的块中恢复缺失的块，而对部分、对象和场景的理解很少。 为了克服这种差异并鼓励学习有用的特征，我们证明了一种简单的策略在计算机视觉中效果很好：掩盖很大一部分随机patch。 这种策略在很大程度上减少了冗余并创建了一项具有挑战性的自我监督任务，需要超越低级图像统计的整体理解。 要对我们的重建任务有一个定性的认识，请参见图 2-4。

**(iii)** The autoencoder’s decoder, which maps the latent representation back to the input, plays a different role between reconstructing text and images. In vision, the decoder reconstructs pixels, hence its output is of a lower semantic level than common recognition tasks. This is in contrast to language, where the decoder predicts missing words that contain rich semantic information. While in BERT the decoder can be trivial (an MLP) [14], we found that for images, the decoder design plays a key role in determining the semantic level of the learned latent representations.

**(iii)** 自动编码器的解码器将潜在表示映射回输入，在重建文本和图像之间扮演不同的角色。 在视觉中，解码器重建像素，因此其输出的语义级别低于常见的识别任务。 这与语言形成对比，在语言中，解码器预测包含丰富语义信息的缺失词。 虽然在 BERT 中，解码器可能很简单（一个 MLP）[14]，但我们发现对于图像，解码器设计在确定学习的潜在表示的语义级别方面起着关键作用。

Driven by this analysis, we present a simple, effective, and scalable form of a masked autoencoder (MAE) for visual representation learning. Our MAE masks random patches from the input image and reconstructs the missing patches in the pixel space. It has an asymmetric encoder-decoder design. Our encoder operates only on the visible subset of patches (without mask tokens), and our decoder is lightweight and reconstructs the input from the latent representation along with mask tokens (Figure 1). Shifting the mask tokens to the small decoder in our asymmetric encoder-decoder results in a large reduction in computation. Under this design, a very high masking ratio (e.g., 75%) can achieve a win-win scenario: it optimizes accuracy while allowing the encoder to process only a small portion (e.g., 25%) of patches. This can reduce overall pre-training time by 3 or more and likewise reduce memory consumption, enabling us to easily scale our MAE to large models.

在此分析的推动下，我们提出了一种简单、有效且可扩展的带掩码自编码器 (MAE) 形式，用于视觉表示学习。 我们的 MAE 从输入图像中屏蔽随机patch，并在像素空间中重建缺失的patch。 它具有不对称的编码器-解码器设计[非对称，编码器看到的和解码器不一样]。 我们的编码器只对可见的patch子集（没有掩码标记）进行操作，我们的解码器是轻量级的，可以从潜在表示中重构输入以及掩码标记（图 1）。 在我们的非对称编码器-解码器中将掩码词元转移到小型解码器会导致计算量大大减少。 在这种设计下，非常高的掩蔽率（例如，75%）可以实现双赢：它优化了准确性，同时允许编码器只处理一小部分（例如，25%）的patch。 这可以将整体预训练时间减少 3 或更多，同时减少内存消耗，使我们能够轻松地将 MAE 扩展到大型模型。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%201.png"/></div>

Figure 1. **Our MAE architecture.** During pre-training, a large random subset of image patches (e.g., 75%) is masked out. The encoder is applied to the small subset of visible patches. Mask tokens are introduced after the encoder, and the full set of encoded patches and mask tokens is processed by a small decoder that reconstructs the original image in pixels. After pre-training, the decoder is discarded and the encoder is applied to uncorrupted images (full sets of patches) for recognition tasks.

图 1. **我们的 MAE 架构。** 在预训练期间，图像块的大量随机子集（例如 75%）被屏蔽掉。 编码器应用于可见patch的小子集。 在编码器之后引入掩码标记，完整的编码patch集和掩码标记由一个小型解码器处理，该解码器以像素为单位重建原始图像。 预训练后，解码器被丢弃，编码器应用于未损坏的图像（完整的patch集）以进行识别任务。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%202.png"/></div>

Figure 2. Example results on ImageNet validation images. For each triplet, we show the masked image (left), our MAE reconstruction† (middle), and the ground-truth (right). The masking ratio is 80%, leaving only 39 out of 196 patches. More examples are in the appendix. †*As no loss is computed on visible patches, the model output on visible patches is qualitatively worse. One can simply overlay the output with the visible patches to improve visual quality. We intentionally opt not to do this, so we can more comprehensively demonstrate the method’s behavior.*
图 2. ImageNet 验证图像的示例结果。 对于每个三元组，我们展示了蒙版图像（左）、我们的 MAE 重建†（中）和真实实况（右）。 掩蔽率为 80%，196 个色块中只剩下 39 个。 更多示例在附录中。 †*由于在可见块上没有计算损失，可见块上的模型输出质量较差。 人们可以简单地用可见的patch覆盖输出以提高视觉质量。 我们有意选择不这样做，因此我们可以更全面地展示该方法的行为。*

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%203.png"/></div>

Figure 3. Example results on COCO validation images, using an MAE trained on ImageNet (the same model weights as in Figure 2). Observe the reconstructions on the two right-most examples, which, although different from the ground truth, are semantically plausible.
图 3. COCO 验证图像的示例结果，使用在 ImageNet 上训练的 MAE（与图 2 中的模型权重相同）。 观察最右边两个例子的重建，虽然与基准不同，但在语义上是合理的。

Our MAE learns very high-capacity models that generalize well. With MAE pre-training, we can train data- hungry models like ViT-Large/-Huge [16] on ImageNet-1K with improved generalization performance. With a vanilla ViT-Huge model, we achieve 87.8% accuracy when fine-tuned on ImageNet-1K. This outperforms all previous results that use only ImageNet-1K data. We also evaluate transfer learning on object detection, instance segmentation, and semantic segmentation. In these tasks, our pre-training achieves better results than its supervised pre-training counterparts, and more importantly, we observe significant gains by scaling up models. These observations are aligned with those witnessed in self-supervised pre-training in NLP [14,47,48,4] and we hope that they will enable our field to explore a similar trajectory.

我们的 MAE 学习了泛化能力非常高的模型。 通过 MAE 预训练，我们可以在 ImageNet-1K 上训练像 ViT-Large/-Huge [16] 这样的数据饥渴模型，并提高泛化性能。 使用 vanilla ViT-Huge 模型，我们在 ImageNet-1K 上进行微调时达到 87.8% 的准确率。 这优于之前仅使用 ImageNet-1K 数据的所有结果。 我们还评估了对象检测、实例分割和语义分割的迁移学习。 在这些任务中，我们的预训练比有监督的预训练取得了更好的结果，更重要的是，我们通过扩大模型观察到了显着的收益。 这些观察结果与 NLP [14,47,48,4] 中自我监督预训练中的观察结果一致，我们希望它们能让我们的领域探索类似的轨迹。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%204.png"/></div>

Figure 4. Reconstructions of ImageNet *validation* images using an MAE pre-trained with a masking ratio of 75% but applied on inputs with higher masking ratios. The predictions differ plausibly from the original images, showing that the method can generalize.
图 4. ImageNet *validation* 图像的重建，使用 MAE 进行了预训练，掩蔽率为 75%，但应用于具有更高掩蔽率的输入。 预测与原始图像似乎有很大不同，表明该方法可以泛化。

## 2. Related Work 相关工作

**Masked language modeling** and its auto-regressive counterparts, e.g., BERT [14] and GPT [47, 48, 4], are highly successful methods for pre-training in NLP. These methods hold out a portion of the input sequence and train models to predict the missing content. These methods have been shown to scale excellently [4] and a large abundance of evidence indicates that these pre-trained representations generalize well to various downstream tasks.

**Masked Language Modeling** 及其自回归模型，例如 BERT [14] 和 GPT [47, 48, 4]，是 NLP 预训练的非常成功的方法。 这些方法保留了输入序列的一部分并训练模型来预测丢失的内容。 这些方法已被证明可以很好地扩展[4]，并且大量证据表明这些预训练的表示可以很好地推广到各种下游任务。

**Autoencoding** is a classical method for learning representations. It has an encoder that maps an input to a latent representation and a decoder that reconstructs the input. For ex-ample, PCA and k-means are autoencoders [29]. Denoising autoencoders (DAE) [58] are a class of autoencoders that corrupt an input signal and learn to reconstruct the original, uncorrupted signal. A series of methods can be thought of as a generalized DAE under different corruptions, e.g., masking pixels [59, 46, 6] or removing color channels [70]. Our MAE is a form of denoising autoencoding, but different from the classical DAE in numerous ways.

**自动编码**是学习表示的经典方法。 它有一个将输入映射到潜在表示的编码器和一个重构输入的解码器。 例如，PCA 和 k-means 是自动编码器 [29]。 去噪自动编码器 (DAE) [58] 是一类自动编码器，它破坏输入信号并学习重建原始的、未破坏的信号。 一系列方法可以被认为是不同损坏下的广义 DAE，例如，屏蔽像素 [59、46、6] 或移除颜色通道 [70]。 我们的 MAE 是一种去噪自编码的形式，但在许多方面与经典 DAE 不同。

**Masked image encoding** methods learn representations from images corrupted by masking. The pioneering work of [59] presents masking as a noise type in DAE. Context Encoder [46] inpaints large missing regions using convolutional networks. Motivated by the success in NLP, related recent methods [6,16,2] are based on Transformers [57]. iGPT [6] operates on sequences of pixels and predicts unknown pixels. The ViT paper [16] studies masked patch prediction for self-supervised learning. Most recently, BEiT [2] proposes to predict discrete tokens [44,50].

**掩码图像编码** [掩码也可以称作蒙版]方法从被掩码损坏的图像中学习表示。 [59] 的开创性工作将掩蔽作为 DAE 中的一种噪声类型。 上下文编码器 [46] 使用卷积网络修复大的缺失区域。 受 NLP 成功的推动，最近的相关方法 [6,16,2] 基于 Transformers [57]。 iGPT [6] 对像素序列进行操作并预测未知像素。 ViT 论文 [16] 研究了用于自我监督学习的掩蔽patch预测。 最近，BEiT [2] 提出预测离散词元 [44,50]。

**Self-supervised learning** approaches have seen significant interest in computer vision, often focusing on different pretext tasks for pre-training [15,61,42,70,45,17]. Recently, contrastive learning [3,22] has been popular, e.g., [62,43,23,7], which models image similarity and dissimilarity (or only similarity [21,8]) between two or more views. Contrastive and related methods strongly depend on data augmentation [7,21,8]. Autoencoding pursues a conceptually different direction, and it exhibits different behaviors as we will present.

**自我监督学习**方法已经引起了计算机视觉的极大兴趣，通常专注于预训练的不同网络前置任务 [15,61,42,70,45,17]。 最近，对比学习 [3,22] 很流行，例如 [62,43,23,7]，它模拟两个或多个视图之间的图像相似性和不相似性（或仅相似性 [21,8]）。 对比和相关方法强烈依赖于数据增强 [7,21,8]。 自编码追求一个概念上不同的方向，它表现出我们将要呈现的不同行为。

## 3. Approach

Our masked autoencoder (MAE) is a simple autoencoding approach that reconstructs the original signal given its partial observation. Like all autoencoders, our approach has an encoder that maps the observed signal to a latent representation, and a decoder that reconstructs the original signal from the latent representation. Unlike classical autoencoders, we adopt an asymmetric design that allows the encoder to operate only on the partial, observed signal (without mask tokens) and a lightweight decoder that re-constructs the full signal from the latent representation and mask tokens. Figure 1 illustrates the idea, introduced next.

我们的掩码自动编码器 (MAE) 是一种简单的自动编码方法，可根据部分观察重建原始信号。 像所有自动编码器一样，我们的方法有一个将观察到的信号映射到潜在表示的编码器[在语义空间上的一个表示]，以及一个从潜在表示重建原始信号的解码器。 与经典的自动编码器不同，我们采用非对称设计，允许编码器仅对部分观察到的信号（没有掩码令牌）进行操作，并采用轻量级解码器从潜在表示和掩码词元中重建完整信号。 图 1 说明了这个想法，接下来介绍。

**Masking.** Following ViT [16], we divide an image into regular non-overlapping patches. Then we sample a subset of patches and mask (i.e., remove) the remaining ones. Our sampling strategy is straightforward: we sample random patches without replacement, following a uniform distribution. We simply refer to this as “random sampling”.

**Masking.** 在 ViT [16] 之后，我们将图像划分为规则的非重叠patch[块]。 然后我们对一个parch 子集进行采样并屏蔽（即删除）剩余的patch。 我们的抽样策略很简单：我们按照均匀分布对随机patch进行抽样而不进行替换。 我们简单地将其称为“随机抽样”。

Random sampling with a high masking ratio (i.e., the ratio of removed patches) largely eliminates redundancy, thus creating a task that cannot be easily solved by extrapolation from visible neighboring patches (see Figures 2 - 4). The uniform distribution prevents a potential center bias (i.e., more masked patches near the image center). Finally, the highly sparse input creates an opportunity for designing an efficient encoder, introduced next.

具有高掩蔽率（即移除补patch的比率）的随机采样在很大程度上消除了冗余，因此创建了一项无法通过从可见的相邻patch外推来轻松解决的任务（参见图 2-4）。 均匀分布可防止潜在的中心偏差（即图像中心附近有更多的蒙版patch）。 最后，高度稀疏的输入为设计高效编码器创造了机会，接下来介绍。

**MAE encoder.** Our encoder is a ViT [16] but applied only on visible, unmasked patches. Just as in a standard ViT, our encoder embeds patches by a linear projection with added positional embeddings, and then processes the resulting set via a series of Transformer blocks. However, our encoder only operates on a small subset (e.g., 25%) of the full set. Masked patches are removed; no mask tokens are used. This allows us to train very large encoders with only a fraction of compute and memory. The full set is handled by a lightweight decoder, described next.

**MAE编码器**。 我们的编码器是 ViT [16]，但仅应用于可见的、未屏蔽的patch。 就像在标准 ViT 中一样，我们的编码器对每个patch进行线性投影并添加位置嵌入，然后通过一系列 Transformer 块处理结果集。 然而，我们的编码器只对整个集合的一小部分（例如 25%）进行操作。 被屏蔽的patch被移除； 不使用掩码标记。 这使我们能够只用一小部分计算和内存来训练非常大的编码器。 全套由轻量级解码器处理，如下所述。

**MAE decoder.** The input to the MAE decoder is the full set of tokens consisting of (i) encoded visible patches, and (ii) mask tokens. See Figure 1. Each mask token [14] is a shared, learned vector that indicates the presence of a missing patch to be predicted. We add positional embeddings to all tokens in this full set; without this, mask tokens would have no information about their location in the image. The decoder has another series of Transformer blocks.

**MAE 解码器。** MAE 解码器的输入是完整的标记集，包括 (i) 编码可见patch[已经通过编码器进行了潜在的表示]和 (ii) 掩码标记。 参见图 1。每个掩码标记 [14] 是一个共享的学习向量，表示存在要预测的缺失patch。 我们将位置嵌入添加到这个完整集合中的所有标记； 没有这个，掩码标记将没有关于它们在图像中的位置的信息。[解码器也是个transform] 解码器有另一个系列的 Transformer 块。

The MAE decoder is only used during pre-training to perform the image reconstruction task (only the encoder is used to produce image representations for recognition). Therefore, the decoder architecture can be flexibly designed in a manner that is independent of the encoder design. We experiment with very small decoders, narrower and shallower than the encoder. For example, our default decoder has <10% computation per token vs. the encoder. With this asymmetrical design, the full set of tokens are only processed by the lightweight decoder, which significantly reduces pre-training time.

MAE 解码器仅在预训练期间用于执行图像重建任务（仅编码器用于生成用于识别的图像表示）。 因此，解码器架构可以以独立于编码器设计的方式灵活设计。 我们尝试了非常小的解码器，比编码器更窄更浅。 例如，我们的默认解码器与编码器相比，每个词元的计算量 <10%。 采用这种非对称设计，全套词元仅由轻量级解码器处理，大大减少了预训练时间。

**Reconstruction target.** Our MAE reconstructs the input by predicting the pixel values for each masked patch. Each element in the decoder’s output is a vector of pixel values representing a patch. The last layer of the decoder is a linear projection whose number of output channels equals the number of pixel values in a patch. The decoder’s output is reshaped to form a reconstructed image. Our loss function computes the mean squared error (MSE) between the reconstructed and original images in the pixel space. We compute the loss only on masked patches, similar to BERT [14]$^1$.

**重建目标。**我们的 MAE 通过预测每个蒙面补丁的像素值来重建输入。 解码器输出中的每个元素都是代表patch的像素值向量。 解码器的最后一层是线性投影，其输出通道数等于patch中像素值的数量。 解码器的输出被重新整形以形成重建图像。 我们的损失函数计算像素空间中重建图像和原始图像之间的均方误差 (MSE)。 我们只计算掩码patch上的损失，类似于 BERT [14]$^1$。

>$^1$ Computing the loss only on masked patches differs from traditional denoising autoencoders [58] that compute the loss on all pixels. This choice is purely result-driven: computing the loss on all pixels leads to a slight decrease in accuracy (e.g., ∼0.5%)

We also study a variant whose reconstruction target is the normalized pixel values of each masked patch. Specifically, we compute the mean and standard deviation of all pixels in a patch and use them to normalize this patch. Using normalized pixels as the reconstruction target improves representation quality in our experiments.

我们还研究了一种变体，其重建目标是每个掩码patch的归一化像素值。 具体来说，我们计算一个patch中所有像素的平均值和标准偏差，并使用它们来规范化这个patch。 在我们的实验中，使用归一化像素作为重建目标可以提高表示质量。

**Simple implementation.** Our MAE pre-training can be implemented efficiently, and importantly, does not require any specialized sparse operations. First we generate a token for every input patch (by linear projection with an added positional embedding). Next we randomly shuffle the list of tokens and remove the last portion of the list, based on the masking ratio. This process produces a small subset of tokens for the encoder and is equivalent to sampling patches without replacement. After encoding, we append a list of mask tokens to the list of encoded patches, and unshuffle this full list (inverting the random shuffle operation) to align all tokens with their targets. The decoder is applied to this full list (with positional embeddings added). As noted, no sparse operations are needed. This simple implementation introduces negligible overhead as the shuffling and unshuffling operations are fast.

简单的实现。 我们的 MAE 预训练可以有效地实施，重要的是，不需要任何专门的稀疏操作。 首先，我们为每个输入patch生成一个标记（通过添加位置嵌入的线性投影）。 接下来，我们根据掩码率随机打乱词元[token]列表并删除列表的最后一部分。 此过程为编码器生成一小部分词元，相当于在不替换的情况下对patch进行采样。 编码后，我们将一个掩码标记列表附加到编码patch列表中，并取消打乱这个完整列表（反转随机打乱操作）以将所有标记与其目标对齐。 解码器应用于这个完整列表（添加了位置嵌入）。 如前所述，不需要稀疏操作。 这个简单的实现引入了可忽略的开销，因为混洗和非混洗操作很快。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%205.png"/></div>

Figure 5. **Masking ratio.** A high masking ratio (75%) works well for both fine-tuning (top) and linear probing (bottom). The y-axes are ImageNet-1K validation accuracy (%) in all plots in this paper.

## 4. ImageNet Experiments ImageNet实验

We do self-supervised pre-training on the ImageNet-1K (IN1K) [13] training set. Then we do supervised training to evaluate the representations with (i) end-to-end fine-tuning or (ii) linear probing. We report top-1 validation accuracy of a single $224 \times 224 $ crop. Details are in Appendix A.1.

我们在 ImageNet-1K (IN1K) [13] 训练集上进行自我监督预训练。 然后我们进行监督训练，通过 (i) 端到端微调[允许改模型中所有可以学习的参数]或 (ii) 线性探测来评估表示[只允许改最后一个全连接的输出]。 我们报告单次 $224  \times 224 $作物的 top-1 验证准确度。 详细信息在附录 A.1 中。

**Baseline: ViT-Large.** We use ViT-Large (ViT-L/16) [16] as the backbone in our ablation study. ViT-L is very big (an order of magnitude bigger than ResNet-50 [25]) and tends to overfit. The following is a comparison between ViT-L trained from scratch vs. fine-tuned from our baseline MAE:

**基准：ViT-Large。** 我们使用 ViT-Large (ViT-L/16) [16] 作为消融研究的主干。 ViT-L 非常大（比 ResNet-50 [25] 大一个数量级）并且容易过拟合。 以下是从头开始训练的 ViT-L 与从我们的基准 MAE 微调的对比：

|scratch, original [16]| scratch, our impl.| baseline MAE|
|---|---|---|
|76.5| 82.5| 84.9|

We note that it is nontrivial to train *supervised* ViT-L from scratch and a good recipe with strong regularization is needed (82.5%, see Appendix A.2). Even so, our MAE pre-training contributes a big improvement. Here fine-tuning is only for 50 epochs (vs. 200 from scratch), implying that the fine-tuning accuracy heavily depends on pre-training.

我们注意到，从头开始训练*监督* ViT-L 是非常重要的，并且需要具有强正则化（82.5%，参见附录 A.2）。 尽管如此，我们的 MAE 预训练还是做出了很大的改进。 这里微调仅针对 50 个 epoch（而从头开始为 200 个），这意味着微调的准确性在很大程度上取决于预训练。

### 4.1. Main Properties 主要结果

We ablate our MAE using the default settings in Table 1 (see caption). Several intriguing properties are observed.

我们使用表 1 中的默认设置消融我们的 MAE（见标题）。 观察到几个有趣的特性。

**Masking ratio**. Figure 5 shows the influence of the masking ratio. The optimal ratios are surprisingly high. The ratio of 75% is good for both linear probing and fine-tuning. This behavior is in contrast with BERT [14], whose typical masking ratio is 15%. Our masking ratios are also much higher than those in related works [6, 16, 2] in computer vision (20% to 50%).

The model infers missing patches to produce different, yet plausible, outputs (Figure 4). It makes sense of the gestalt of objects and scenes, which cannot be simply completed by extending lines or textures. We hypothesize that this reasoning-like behavior is linked to the learning of useful representations.

Figure 5 also shows that linear probing and fine-tuning results follow *different* trends. For linear probing, the accuracy increases steadily with the masking ratio until the sweet point: the accuracy gap is up to 20% (54.6% vs. 73.5%). For fine-tuning, the results are less sensitive to the ratios, and a wide range of masking ratios (40-80%) work well. All fine-tuning results in Figure 5 are better than training from scratch (82.5%).

Decoder design. Our MAE decoder can be flexibly designed, as studied in Table 1a and 1b.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%201.png"/></div>

Table 1. MAE ablation experiments with ViT-L/16 on ImageNet-1K. We report fine-tuning (ft) and linear probing (lin) accuracy (%). If not specified, the default is: the decoder has depth 8 and width 512, the reconstruction target is unnormalized pixels, the data augmentation is random resized cropping, the masking ratio is 75%, and the pre-training length is 800 epochs. Default settings are marked in `gray`.
表 1. 在 ImageNet-1K 上使用 ViT-L/16 进行 MAE 消融实验。 我们报告微调 (ft) 和线性探测 (lin) 精度 (%)。 如果不指定，默认为：解码器深度为8，宽度为512，重建目标为非归一化像素，数据增强为随机调整大小裁剪，掩蔽率为75%，预训练长度为800 epochs。 默认设置以“灰色”标记。

Table 1a varies the decoder depth (number of Transformer blocks). A sufficiently deep decoder is important for linear probing. This can be explained by the gap between a pixel reconstruction task and a recognition task: the last several layers in an autoencoder are more specialized for reconstruction, but are less relevant for recognition. A reasonably deep decoder can account for the reconstruction specialization, leaving the latent representations at a more abstract level. This design can yield up to 8% improvement in linear probing (Table 1a, ‘lin’). However, if fine-tuning is used, the last layers of the encoder can be tuned to adapt to the recognition task. The decoder depth is less influential for improving fine-tuning (Table 1a, ‘ft’).

Interestingly, our MAE with a single-block decoder can perform strongly with fine-tuning (84.8%). Note that a single Transformer block is the minimal requirement to propagate information from visible tokens to mask tokens. Such a small decoder can further speed up training.

In Table 1b we study the decoder width (number of chan-nels). We use 512-d by default, which performs well under fine-tuning and linear probing. A narrower decoder also works well with fine-tuning.

Overall, our default MAE decoder is lightweight. It has 8 blocks and a width of 512-d ( `gray` in Table 1). It only has 9% FLOPs per token vs. ViT-L (24 blocks, 1024-d). As such, while the decoder processes all tokens, it is still a small fraction of the overall compute.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%202.png"/></div>

Table 2. **Wall-clock time** of our MAE training (800 epochs), benchmarked in 128 TPU-v3 cores with TensorFlow. The speedup is relative to the entry whose encoder has mask tokens (gray). The decoder width is 512, and the mask ratio is 75%. y: This entry is estimated by training ten epochs.
表 2. 我们的 MAE 训练（800 个 epoch）的**执行时间**，在 128 个 TPU-v3 内核和 TensorFlow 中进行了基准测试。 加速与编码器具有掩码令牌（灰色）的条目相关。 解码器宽度为512，掩码率为75%。 y：这个条目是通过训练十个 epoch 来估计的。

**Mask token.** An important design of our MAE is to skip the mask token [M] in the encoder and apply it later in the lightweight decoder. Table 1c studies this design.

If the encoder uses mask tokens, it performs worse: its accuracy drops by 14% in linear probing. In this case, there is a gap between pre-training and deploying: this encoder has a large portion of mask tokens in its input in pretraining, which does not exist in uncorrupted images. This gap may degrade accuracy in deployment. By removing the mask token from the encoder, we constrain the encoder to always see real patches and thus improve accuracy.

Moreover, by skipping the mask token in the encoder, we greatly reduce training computation. In Table 1c, we reduce the overall training FLOPs by 3.3. This leads to a 2.8 wall-clock speedup in our implementation (see Table 2). The wall-clock speedup is even bigger (3.5-4.1), for a smaller decoder (1-block), a larger encoder (ViT-H), or both. Note that the speedup can be >4 for a masking ratio of 75%, partially because the self-attention complexity is quadratic. In addition, memory is greatly reduced, which can enable training even larger models or speeding up more by large-batch training. The time and memory efficiency makes our MAE favorable for training very large models. 

**Reconstruction target.** We compare different reconstruction targets in Table 1d. Our results thus far are based on pixels without (per-patch) normalization. Using pixels with normalization improves accuracy. This per-patch normalization enhances the contrast locally. In another variant, we perform PCA in the patch space and use the largest PCA coefficients (96 here) as the target. Doing so degrades ac-curacy. Both experiments suggest that the high-frequency components are useful in our method.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%206.png"/></div>

Figure 6. **Mask sampling strategies** determine the pretext task difficulty, influencing reconstruction quality and representations (Table 1f). Here each output is from an MAE trained with the specified masking strategy. Left: random sampling (our default). Middle: block-wise sampling [2] that removes large random blocks.Right: grid-wise sampling that keeps one of every four patches.Images are from the validation set.
图 6. **Mask sampling strategy** 确定下游任务难度，影响重建质量和表示（表 1f）。 这里的每个输出都来自使用指定掩蔽策略训练的 MAE。 左：随机抽样（我们的默认设置）。 中间：块状采样[2]，删除大的随机块。右：网格采样，保留每四个补丁之一。图像来自验证集。

We also compare an MAE variant that predicts tokens, the target used in BEiT [2]. Specifically for this variant, we use the DALLE pre-trained dVAE [50] as the tokenizer, following [2]. Here the MAE decoder predicts the token indices using cross-entropy loss. This tokenization improves fine-tuning accuracy by 0.4% vs. unnormalized pixels, but has no advantage vs. normalized pixels. It also reduces linear probing accuracy. In §5 we further show that tokenization is not necessary in transfer learning.

Our *pixel*-based MAE is much simpler than tokenization. The dVAE tokenizer requires one more pre-training stage, which may depend on extra data (250M images [50]). The dVAE encoder is a large convolutional network (40% FLOPs of ViT-L) and adds nontrivial overhead. Using pixels does not suffer from these problems.

Data augmentation. Table 1e studies the influence of data augmentation on our MAE pre-training.

Our MAE works well using cropping-only augmentation, either fixed-size or random-size (both having random horizontal flipping). Adding color jittering degrades the results and so we do not use it in other experiments.

Surprisingly, our MAE behaves decently even if using no data augmentation (only center-crop, no flipping). This property is dramatically different from contrastive learning and related methods [62,23,7,21], which heavily rely on data augmentation. It was observed [21] that using cropping-only augmentation reduces the accuracy by 13% and 28% respectively for BYOL [21] and SimCLR [7]. In addition, there is no evidence that contrastive learning can work without augmentation: the two views of an image are the same and can easily satisfy a trivial solution.

In MAE, the role of data augmentation is mainly per-formed by random masking (ablated next). The masks are different for each iteration and so they generate new training samples regardless of data augmentation. The pretext task is made difficult by masking and requires less augmentation to regularize training.

**Mask sampling strategy.** In Table 1f we compare different mask sampling strategies, illustrated in Figure 6.

The block-wise masking strategy, proposed in [2], tends to remove large blocks (Figure 6 middle). Our MAE with block-wise masking works reasonably well at a ratio of 50%, but degrades at a ratio of 75%. This task is harder than that of random sampling, as a higher training loss is observed. The reconstruction is also blurrier.

We also study *grid-wise* sampling, which regularly keeps one of every four patches (Figure 6 right). This is an easier task and has lower training loss. The reconstruction is sharper. However, the representation quality is lower.
Simple random sampling works the best for our MAE. It allows for a higher masking ratio, which provides a greater speedup benefit while also enjoying good accuracy.

**Training schedule.** Our ablations thus far are based on 800-epoch pre-training. Figure 7 shows the influence of the training schedule length. The accuracy improves steadily with longer training. Indeed, we have not observed saturation of linear probing accuracy even at 1600 epochs. This behavior is unlike contrastive learning methods, e.g., MoCo v3 [9] saturates at 300 epochs for ViT-L. Note that the MAE encoder only sees 25% of patches per epoch, while in contrastive learning the encoder sees 200% (two- crop) or even more (multi-crop) patches per epoch.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%207.png"/></div>

Figure 7. Training schedules. A longer training schedule gives a noticeable improvement. Here each point is a full training schedule. The model is ViT-L with the default setting in Table 1.
图 7. 培训时间表。 更长的训练计划会带来明显的改善。 这里的每一点都是一个完整的训练计划。 型号为 ViT-L，默认设置见表 1。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%203.png"/></div>

Table 3. **Comparisons with previous results on ImageNet-1K**. The pre-training data is the ImageNet-1K training set (except the tokenizer in BEiT was pre-trained on 250M DALLE data [50]). All self-supervised methods are evaluated by end-to-end fine-tuning. The ViT models are B/16, L/16, H/14 [16]. The best for each column is underlined. All results are on an image size of 224, except for ViT-H with an extra result on 448. Here our MAE reconstructs normalized pixels and is pre-trained for 1600 epochs.
表 3. **与 ImageNet-1K 上的先前结果的比较**。 预训练数据是 ImageNet-1K 训练集（除了 BEiT 中的分词器是在 250M DALLE 数据上预训练的 [50]）。 所有自监督方法都通过端到端微调进行评估。 ViT 型号为 B/16、L/16、H/14 [16]。 每列的最佳值带有下划线。 所有结果都在 224 的图像大小上，除了 ViT-H 在 448 上的额外结果。这里我们的 MAE 重建归一化像素并预训练了 1600 个 epoch。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%208.png"/></div>

Figure 8. **MAE pre-training vs. supervised pre-training**, evaluated by fine-tuning in ImageNet-1K (224 size). We compare with the original ViT results [16] trained in IN1K or JFT300M.
图 8. **MAE 预训练与监督预训练**，通过 ImageNet-1K（224 大小）中的微调进行评估。 我们与在 IN1K 或 JFT300M 中训练的原始 ViT 结果 [16] 进行比较。

### 4.2. Comparisons with Previous Results

Comparisons with self-supervised methods. In Table 3 we compare the fine-tuning results of self-supervised ViT models. For ViT-B, all methods perform closely. For ViT-L, the gaps among methods are bigger, suggesting that a challenge for bigger models is to reduce overfitting.

Our MAE can scale up easily and has shown steady im-provement from bigger models. We obtain 86.9% accuracy using ViT-H (224 size). By fine-tuning with a 448 size, we achieve **87.8%** accuracy, using only IN1K data. The previous best accuracy, among all methods using only IN1K data, is 87.1% (512 size) [67], based on advanced networks. We improve over the state-of-the-art by a nontrivial margin in the highly competitive benchmark of IN1K (no external data). Our result is based on vanilla ViT, and we expect advanced networks will perform better.

Comparing with BEiT [2], our MAE is more accurate while being simpler and faster. Our method reconstructs pixels, in contrast to BEiT that predicts tokens: BEiT reported a 1.8% degradation [2] when reconstructing pixels with ViT-B$^2$.  We do not need dVAE pre-training. Moreover, our MAE is considerably faster (3.5 per epoch) than BEiT, for the reason as studied in Table 1c.

>$^2$ We observed the degradation also in BEiT with ViT-L: it produces 85.2% (tokens) and 83.5% (pixels), reproduced from the official code.

The MAE models in Table 3 are pre-trained for 1600 epochs for better accuracy (Figure 7). Even so, our total pre-training time is less than the other methods when trained on the same hardware. For example, training ViT-L on 128 TPU-v3 cores, our MAE’s training time is 31 hours for 1600 epochs and MoCo v3’s is 36 hours for 300 epochs [9].

Comparisons with supervised pre-training. In the original ViT paper [16], ViT-L degrades when trained in IN1K. Our implementation of supervised training (see A.2) works better, but accuracy saturates. See Figure 8.
Our MAE pre-training, using only IN1K, can generalize better: the gain over training from scratch is bigger for higher-capacity models. It follows a trend similar to the JFT-300M supervised pre-training in [16]. This comparison shows that our MAE can help scale up model sizes.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Fig%209.png"/></div>

Figure 9. **Partial fine-tuning** results of ViT-L w.r.t. the number of fine-tuned Transformer blocks under the default settings from Table 1. Tuning 0 blocks is linear probing; 24 is full fine-tuning.Our MAE representations are less linearly separable, but are consistently better than MoCo v3 if one or more blocks are tuned.

图 9. ViT-L w.r.t 的**部分微调**结果 在表 1 的默认设置下微调 Transformer 块的数量。调整 0 个块是线性探测； 24 是完全微调。我们的 MAE 表示的线性可分性较差，但如果调整一个或多个块，则始终优于 MoCo v3。

### 4.3. Partial Fine-tuning

Table 1 shows that linear probing and fine-tuning results are largely uncorrelated. Linear probing has been a popular protocol in the past few years; however, it misses the opportunity of pursuing strong but non-linear features—which is indeed a strength of deep learning. As a middle ground, we study a partial fine-tuning protocol: fine-tune the last several layers while freezing the others. This protocol was also used in early works, e.g., [65,70,42].

表 1 显示线性探测和微调结果在很大程度上是不相关的。 线性探测在过去几年中一直是一种流行的协议。 然而，它错过了追求强大但非线性特征的机会——这确实是深度学习的一个优势。 作为中间立场，我们研究了一种部分微调协议：微调最后几层，同时冻结其他层。 该协议也用于早期工作，例如 [65,70,42]。

Figure 9 shows the results. Notably, fine-tuning only one Transformer block boosts the accuracy significantly from 73.5% to 81.0%. Moreover, ifwe fine-tune only “half” of the last block (i.e., its MLP sub-block), we can get 79.1%, much better than linear probing. This variant is essentially fine-tuning an MLP head. Fine-tuning a few blocks (e.g., 4 or 6) can achieve accuracy close to full fine-tuning.

In Figure 9 we also compare with MoCo v3 [9], a contrastive method with ViT-L results available. MoCo v3 has higher linear probing accuracy; however, all of its partial fine-tuning results are worse than MAE. The gap is 2.6% when tuning 4 blocks. While the MAE representations are less linearly separable, they are stronger non-linear features and perform well when a non-linear head is tuned.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%204.png"/></div>

Table 4. **COCO object detection and segmentation** using a ViT Mask R-CNN baseline. All entries are based on our implementation. Self-supervised entries use IN1K data without labels. Mask AP follows a similar trend as box AP.

These observations suggest that linear separability is not the sole metric for evaluating representation quality. It has also been observed (e.g., [8]) that linear probing is not well correlated with transfer learning performance, e.g., for object detection. To our knowledge, linear evaluation is not often used in NLP for benchmarking pre-training.

## 5. Transfer Learning Experiments 迁移学习实验

We evaluate transfer learning in downstream tasks using the pre-trained models in Table 3.

Object detection and segmentation. We fine-tune Mask R-CNN [24] end-to-end on COCO [37]. The ViT backbone is adapted for use with FPN [36] (see A.3). We apply this approach for all entries in Table 4. We report box AP for object detection and mask AP for instance segmentation.

Compared to supervised pre-training, our MAE performs better under all configurations (Table 4). With the smaller ViT-B, our MAE is 2.4 points higher than supervised pre-training (50.3 vs. 47.9, APbox). More significantly, with the larger ViT-L, our MAE pre-training outperforms supervised pre-training by 4.0 points (53.3 vs. 49.3).

The pixel-based MAE is better than or on par with the token-based BEiT, while MAE is much simpler and faster. Both MAE and BEiT are better than MoCo v3 and MoCo v3 is on par with supervised pre-training.

Semantic segmentation. We experiment on ADE20K [72] using UperNet [63] (see A.4). Table 5 shows that our pretraining significantly improves results over supervised pretraining, e.g., by 3.7 points for ViT-L. Our pixel-based MAE also outperforms the token-based BEiT. These observations are consistent with those in COCO.

Classification tasks. Table 6 studies transfer learning on the iNaturalists [56] and Places [71] tasks (see A.5). On iNat, our method shows strong scaling behavior: accuracy improves considerably with bigger models. Our results surpass the previous best results by large margins. On Places, our MAE outperforms the previous best results [19, 40], which were obtained via pre-training on billions of images.

Pixels vs. tokens. Table 7 compares pixels vs. tokens as the MAE reconstruction target. While using dVAE tokens is better than using unnormalized pixels, it is statistically similar to using normalized pixels across all cases we tested. It again shows that tokenization is not necessary for our MAE.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%205.png"/></div>

Table 5. **ADE20K semantic segmentation** (mIoU) using Uper-Net. BEiT results are reproduced using the official code. Other entries are based on our implementation. Self-supervised entries use IN1K data without labels.  dataset ViT-B ViT-L ViT-H ViT-H448 prev best

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%206.png"/></div>

Table 6. **Transfer learning accuracy on classification datasets**, using MAE pre-trained on IN1K and then fine-tuned. We provide system-level comparisons with the previous best results.
表 6. **在分类数据集上迁移学习准确度**，使用在 IN1K 上预训练然后微调的 MAE。 我们提供与之前最佳结果的系统级比较。
$^†$: pre-trained on 1 billion images. $^‡$: pre-trained on 3.5 billion images.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%207.png"/></div>

Table 7. **Pixels *vs.* tokens** as the MAE reconstruction target. $$ is the difference between using dVAE tokens and using normalized pixels. The difference is statistically insignificant.

## 6. Discussion and Conclusion 讨论和结论

Simple algorithms that scale well are the core of deep learning. In NLP, simple self-supervised learning methods (e.g., [47, 14, 48, 4]) enable benefits from exponentially scaling models. In computer vision, practical pre-training paradigms are dominantly supervised (e.g. [33, 51, 25, 16]) despite progress in self-supervised learning. In this study, we observe on ImageNet and in transfer learning that an autoencoder—a simple self-supervised method similar to techniques in NLP—provides scalable benefits. Self-supervised learning in vision may now be embarking on a similar trajectory as in NLP.

可扩展性良好的简单算法是深度学习的核心。 在 NLP 中，简单的自我监督学习方法（例如，[47,14,48,4]）可以从指数缩放模型中受益。 在计算机视觉中，尽管自监督学习取得了进展，但实际的预训练范式主要受到监督（例如 [33、51、25、16]）。 在这项研究中，我们在 ImageNet 和迁移学习中观察到自动编码器（一种类似于 NLP 技术的简单自我监督方法）提供了可扩展的优势。 视觉中的自我监督学习现在可能走上了与 NLP 类似的轨迹。

On the other hand, we note that images and languages are signals of a different nature and this difference must be addressed carefully. Images are merely recorded light without a semantic decomposition into the visual analogue of words. Instead of attempting to remove objects, we remove random patches that most likely do not form a semantic segment. Likewise, our MAE reconstructs pixels, which are not semantic entities. Nevertheless, we observe (e.g., Figure 4) that our MAE infers complex, holistic reconstructions, suggesting it has learned numerous visual concepts, i.e., semantics. We hypothesize that this behavior occurs by way of a rich hidden representation inside the MAE. We hope this perspective will inspire future work.

另一方面，我们注意到图像和语言是不同性质的信号，必须仔细处理这种差异。 图像只是记录下来的光，没有将语义分解为单词的视觉类似物。 我们没有尝试删除对象，而是删除了最有可能不形成语义段的随机patch。 同样，我们的 MAE 重建不是语义实体的像素。 尽管如此，我们观察到（例如，图 4）我们的 MAE 推断出复杂的整体重建，这表明它已经学习了许多视觉概念，即语义。 我们假设这种行为是通过 MAE 内部丰富的隐藏表示发生的。 我们希望这种观点能够激发未来的工作。

**Broader impacts.** The proposed method predicts content based on learned statistics of the training dataset and as such will reflect biases in those data, including ones with negative societal impacts. The model may generate inexistent content. These issues warrant further research and consideration when building upon this work to generate images.

**更广泛的影响。** 建议的方法根据训练数据集的学习统计数据预测内容，因此将反映这些数据中的偏差，包括具有负面社会影响的偏差。 该模型可能会生成不存在的内容。 在此工作的基础上生成图像时，这些问题值得进一步研究和考虑。

## References 参考文献

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv:1607.06450, 2016.

[2] Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT pre-training of image transformers. arXiv:2106.08254, 2021. Accessed in June 2021.

[3] Suzanna Becker and Geoffrey E Hinton. Self-organizing neural network that discovers surfaces in random-dot stereograms. Nature, 1992.

[4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020.

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

[6] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual rep-resentations. In ICML, 2020.

[8] Xinlei Chen and Kaiming He. Exploring simple Siamese representation learning. In CVPR, 2021.

[9] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised Vision Transformers. In ICCV, 2021.

[10] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. ELECTRA: Pre-training text encoders as discriminators rather than generators. In ICLR, 2020.

[11] Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine learning, 1995.

[12] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Ran-daugment: Practical automated data augmentation with a reduced search space. In CVPR Workshops, 2020.

[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.

[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.

[15] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.

[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa De- hghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[17] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.

[18] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010.

[19] Priya Goyal, Mathilde Caron, Benjamin Lefaudeux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh, Vitaliy Liptchinsky, Is- han Misra, Armand Joulin, and Piotr Bojanowski. Self-supervised pretraining of visual features in the wild. arXiv:2103.01988, 2021.

[20] Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677, 2017.

[21] Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap your own latent - a new approach to self-supervised learning. In NeurIPS, 2020.

[22] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR, 2006.

[23] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Gir- shick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

[24] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask R-CNN. In ICCV, 2017.

[25] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[26] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, 2021.

[27] Dan Hendrycks and Thomas Dietterich. Benchmarking neural net-work robustness to common corruptions and perturbations. In ICLR, 2019.

[28] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In CVPR, 2021.

[29] Geoffrey E Hinton and Richard S Zemel. Autoencoders, minimum description length, and helmholtz free energy. In NeurIPS, 1994.

[30] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Wein-berger. Deep networks with stochastic depth. In ECCV, 2016.

[31] Sergey Ioffe and Christian Szegedy. Batch normalization: Accel-erating deep network training by reducing internal covariate shift. In ICML, 2015.

[32] Insoo Kim, Seungju Han, Ji-won Baek, Seong-Jin Park, Jae-Joon Han, and Jinwoo Shin. Quality-agnostic image recognition via invertible decoder. In CVPR, 2021.

[33] Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet clas-sification with deep convolutional neural networks. In NeurIPS, 2012.

[34] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[35] Yanghao Li, Saining Xie, Xinlei Chen, Piotr Dollar, Kaiming He, and Ross Girshick. Benchmarking detection transfer learning with vision transformers. In preparation, 2021.

[36] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In CVPR, 2017.

[37] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Mi-crosoft COCO: Common objects in context. In ECCV, 2014.

[38] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017.

[39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regu-larization. In ICLR, 2019.

[40] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV, 2018.

[41] Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Ranjie Duan, Shaokai Ye, Yuan He, and Hui Xue. Towards robust vision trans-former. arXiv:2105.07926, 2021.

[42] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016.

[43] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018.

[44] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.

[45] Deepak Pathak, Ross Girshick, Piotr Dollar, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In CVPR, 2017.

[46] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016.

[47] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pretraining. 2018.

[48] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[49] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.

[50] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.

[51] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[52] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[53] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through attention. In ICML, 2021.

[54] Hugo Touvron, Alexandre Sablayrolles, Matthijs Douze, Matthieu Cord, and Herve Jegou. Grafit: Learning fine-grained image repre-sentations with coarse labels. In ICCV, 2021.

[55] Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy. arXiv:1906.06423, 2019.

[56] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Be- longie. The iNaturalist species classification and detection dataset. In CVPR, 2018.

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

[58] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre- Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.

[59] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol, and Leon Bottou. Stacked denoising au-toencoders: Learning useful representations in a deep network with a local denoising criterion. JMLR, 2010.

[60] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In NeurIPS, 2019.

[61] Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In ICCV, 2015.

[62] Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin. Unsuper-vised feature learning via non-parametric instance discrimination. In CVPR, 2018.

[63] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018.

[64] Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dollar, and Ross Girshick. Early convolutions help transformers see better. In NeurIPS, 2021.

[65] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In NeurIPS, 2014.

[66] Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv:1708.03888, 2017.

[67] Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, and Shuicheng Yan. VOLO: Vision outlooker for visual recognition. arXiv:2106.13112, 2021.

[68] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In ICCV, 2019.

[69] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In ICLR, 2018.

[70] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016.

[71] Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. Learning deep features for scene recognition using Places database. In NeurIPS, 2014.

[72] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ADE20K dataset. IJCV, 2019.

## A. Implementation Details

### A.1. ImageNet Experiments

**ViT architecture.** We follow the standard ViT architecture [16]. It has a stack of Transformer blocks [57], and each block consists of a multi-head self-attention block and an MLP block, both having LayerNorm (LN) [1]. The encoder ends with LN. As the MAE encoder and decoder have different width, we adopt a linear projection layer after the encoder to match it. Our MAE adds positional embeddings [57] (the sine-cosine version) to both the encoder and decoder inputs. Our MAE does not use relative position or layer scaling (which are used in the code of [2]).

We extract features from the encoder output for finetuning and linear probing. As ViT has a class token [16], to adapt to this design, in our MAE pre-training we append an auxiliary dummy token to the encoder input. This token will be treated as the class token for training the classifier in linear probing and fine-tuning. Our MAE works similarly well without this token (with average pooling).

**Pre-training.** The default setting is in Table 8. We do not use color jittering, drop path, or gradient clip. We use xavier.uniform [18] to initialize all Transformer blocks, fol-lowing ViT’s official code [16]. We use the linear lr scaling rule [20]: $lr = base\_lr\times batchsize / 256$.

**End-to-end fine-tuning.** Our fine-tuning follows common practice of supervised ViT training. The default setting is in Table 9. We use layer-wise lr decay [10] following [2].

**Linear probing.** Our linear classifier training follows [9]. See Table 10. We observe that linear probing requires a very different recipe than end-to-end fine-tuning. In particular, regularization is in general harmful for linear probing. Fol-lowing [9], we disable many common regularization strategies: we do not use mixup [69], cutmix [68], drop path [30], or color jittering, and we set weight decay as zero.

It is a common practice to normalize the classifier input when training a classical linear classifier (e.g., SVM [11]). Similarly, it is beneficial to normalize the pre-trained features when training the linear probing classifier. Following [15], we adopt an extra BatchNorm layer [31] without affine transformation (`affine=False`). This layer is applied on the pre-trained features produced by the encoder, and is before the linear classifier. We note that the layer does not break the linear property, and it can be absorbed into the linear classifier after training: it is essentially a reparameterized linear classifier$^3$.  Introducing this layer helps calibrate the feature magnitudes across different variants in our ablations, so that they can use the same setting without further $lr$ search.

>$^3$ Alternatively, we can pre-compute the mean and std of the features and use the normalized features to train linear classifiers

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%208.png"/></div>

Table 8. **Pre-training setting.**

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%209.png"/></div>

Table 9. **End-to-end fine-tuning setting.**

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%2010.png"/></div>

Table 10. **Linear probing setting.** We use LARS with a large batch for faster training; SGD works similarly with a 4096 batch.

**Partial fine-tuning.** Our MAE partial fine-tuning (§4.3) follows the setting in Table 9, except that we adjust the number of fine-tuning epochs. We observe that tuning fewer blocks requires a longer schedule. We set the numbers of fine-tuning epochs as f50, 100, 200g and use the optimal one for each number of blocks tuned.

### A.2. Supervised Training ViT-L/H from Scratch

We find that it is nontrivial to train supervised ViT-L/H from scratch on ImageNet-1K. The training is unstable. While there have been strong baselines with publicly available implementations [53] for smaller models, the recipes for the larger ViT-L/H are unexplored. Directly applying the previous recipes to these larger models does not work. A NaN loss is frequently observed during training.

We provide our recipe in Table 11. We use a wd of 0.3, a large batch size of 4096, and a long warmup, following the original ViT [16]. We use 2=0:95 following [6]. We use the regularizations listed in Table 11 and disable others, following [64]. All these choices are for improving training stability. Our recipe can finish training with no NaN loss.The accuracy is 82.6% for ViT-L (81.5% w/o EMA), and 83.1% for ViT-H (80.9% w/o EMA). Both ViT-L and ViT-H show an overfitting trend if not using EMA.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%2011.png"/></div>

Table 11. **Supervised training ViT from scratch.**

As a by-product, our recipe for ViT-B has 82.3% accuracy (82.1% w/o EMA), vs. 81.8% in [53].

### A.3. Object Detection and Segmentation in COCO

We adapt the vanilla ViT for the use of an FPN backbone [36] in Mask R-CNN [24]. ViT has a stack of Transformer blocks that all produce feature maps at a single scale (e.g., stride 16). We equally divide this stack into 4 subsets and apply convolutions to upsample or downsample the inter-mediate feature maps for producing different scales (stride 4, 8, 16, or 32, the same as a standard ResNet [25]). FPN is built on these multi-scale maps.

For fair comparisons among different methods, we search for hyper-parameters for each entry in Table 4 (including all competitors). The hyper-parameters we search for are the learning rate, weight decay, drop path rate, and fine-tuning epochs. We will release code along with the specific configurations. For full model and training details, plus additional experiments, see [35].

### A.4. Semantic Segmentation in ADE20K

We use UperNet [63] following the semantic segmentation code of [2]. We fine-tune end-to-end for 100 epochs with a batch size of 16. We search for the optimal lr for each entry in Table 5 (including all competitors).

The semantic segmentation code of [2] uses relative po-sition bias [49]. Our MAE pre-training does not use it. For fair comparison, we turn on relative position bias only during transfer learning, initialized as zero. We note that our BEiT reproduction uses relative position bias in both pretraining and fine-tuning, following their code.

### A.5. Additional Classification Tasks

We follow the setting in Table 9 for iNaturalist and Places fine-tuning (Table 6). We adjust the lr and finetuning epochs for each individual dataset.
method model params acc

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%2012.png"/></div>

Table 12. **Linear probing results of masked encoding methods.**

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Table%2013.png"/></div>

Table 13. **Robustness evaluation on ImageNet variants** (top-1 accuracy, except for IN-C [27] which evaluates mean corruption error). We test the same MAE models (Table 3) on different Im- ageNet validation sets, without any specialized fine-tuning. We provide system-level comparisons with the previous best results.

## B. Comparison on Linear Probing Results

In §4.3 we have shown that linear probing accuracy and fine-tuning accuracy are largely uncorrelated and they have different focuses about linear separability. We notice that existing masked image encoding methods are generally less competitive in linear probing (e.g., than contrastive learning). For completeness, in Table 12 we compare on linear probing accuracy with masking-based methods.

Our MAE with ViT-L has 75.8% linear probing accuracy. This is substantially better than previous maskingbased methods. On the other hand, it still lags behind contrastive methods under this protocol: e.g., MoCo v3 [9] has 77.6% linear probing accuracy for the ViT-L (Figure 9).

### C. Robustness Evaluation on ImageNet

In Table 13 we evaluate the robustness of our models on different variants of ImageNet validation sets. We use the same models fine-tuned on original ImageNet (Table 3) and only run inference on the different validation sets, without any specialized fine-tuning. Table 13 shows that our method has strong scaling behavior: increasing the model sizes has significant gains. Increasing the image size helps in all sets but IN-C. Our results outperform the previous best results (of specialized systems) by large margins.
In contrast, supervised training performs much worse (Table 13 bottom; models described in A.2). For example, with ViT-H, our MAE pre-training is 35% better on IN-A (68.2% vs 33.1%) than the supervised counterpart. 

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.png"/></div>

Figure 10. **Uncurated random samples** on ImageNet validation images. For each triplet, we show the masked image (left), our MAE reconstruction (middle), and the ground-truth (right). The masking ratio is 75%. 

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners2.png"/></div>

Figure 11. **Uncurated random samples** on COCO validation images, using an MAE trained on ImageNet. For each triplet, we show the masked image (left), our MAE reconstruction (middle), and the ground-truth (right). The masking ratio is 75%.
