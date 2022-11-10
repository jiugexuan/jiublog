---
title: 【论文】BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding BERT：用于语言理解的深度双向变换器的预训练
date: 2022-10-20 08:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
---

<div align = center>
Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova </div>
<div align = center>Google AI Language</div>
<div align = center>
{jacobdevlin,mingweichang,kentonl,kristout}@google.com
</div>

## Abstract 摘要

We introduce a new language representation model called **BERT**, which stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.

我们引入了一种称为 **BERT** 的新语言表示模型，它代表 **B**idirectional **E**ncoder **R**epresentations from **T**transformers。 与最近的语言表示模型（Peters et al., 2018a; Radford et al., 2018）不同，BERT 旨在通过联合调节所有层的左右上下文，从未标记的文本中预训练深度双向表示。 因此，预训练的 BERT 模型只需一个额外的输出层即可进行微调，从而为各种任务（例如问答和语言推理）创建最先进的模型，而无需对特定于任务的架构进行大量修改。

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

BERT 在概念上很简单，在实验上很强大。 它在 11 个自然语言处理任务上获得了新的 state-of-the-art 结果，包括将 GLUE 分数推至 80.5%（7.7% 点的绝对改进），MultiNLI 准确度达到 86.7%（4.6% 的绝对改进），SQuAD v1.1 问答测试 F1 到 93.2（1.5 分绝对提高）和 SQuAD v2.0 测试 F1 到 83.1（5.1 分绝对提高）。

## 1 Introduction 简介

Language model pre-training has been shown to be effective for improving many natural language processing tasks (Dai and Le, 2015; Peters et al., 2018a; Radford et al., 2018; Howard and Ruder, 2018). These include sentence-level tasks such as natural language inference (Bowman et al., 2015; Williams et al., 2018) and paraphrasing (Dolan and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level (Tjong Kim Sang and De Meulder, 2003; Rajpurkar et al., 2016).

语言模型预训练已被证明可有效改善许多自然语言处理任务（Dai 和 Le，2015；Peters 等人，2018a；Radford 等人，2018；Howard 和 Ruder，2018）。 其中包括句子级任务，例如自然语言推理（Bowman 等人，2015；Williams 等人，2018）和释义（Dolan 和 Brockett，2005），旨在通过整体分析来预测句子之间的关系，以及词元级别的任务，例如命名实体识别和问答，其中需要模型在词元级别产生细粒度的输出（Tjong Kim Sang 和 De Meulder，2003；Rajpurkar 等人，2016）。

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo (Peters et al., 2018a), uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pretrained parameters. The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

将预训练的语言表示应用于下游任务有两种现有策略：基于特征和微调。 基于特征的方法，例如 ELMo (Peters et al., 2018a)，使用特定于任务的架构，其中包括预训练的表示作为额外特征。 微调方法，例如 Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018)，引入了最少的任务特定参数，并通过简单地微调所有预训练参数来对下游任务进行训练。 这两种方法在预训练期间共享相同的目标函数，它们使用单向语言模型来学习通用语言表示。

We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-to- right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying finetuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

我们认为，当前的技术限制了预训练表示的能力，特别是对于微调方法。 主要限制是标准语言模型是单向的，这限制了可在预训练期间使用的架构的选择。 例如，在 OpenAI GPT 中，作者使用从左到右的架构，其中每个标记只能关注 Transformer 的自注意力层中的先前标记（Vaswani 等，2017）。 这样的限制对于句子级任务来说是次优的，并且在将基于微调的方法应用于词元级任务（例如问答）时可能非常有害，在这些任务中，从两个方向整合上下文至关重要。

In this paper, we improve the fine-tuning based approaches by proposing BERT: **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. BERT alleviates the previously mentioned unidirectionality constraint by using a “masked language model” (MLM) pre-training objective, inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also use a “next sentence prediction” task that jointly pre-trains text-pair representations. The contributions of our paper are as follows:

在本文中，我们通过提出 BERT 改进了基于微调的方法：**B**idirectional **E**ncoder **R**epresentations from **T**transformers。 BERT 通过使用受完形填空任务 (Taylor, 1953) 启发的“掩码语言模型” (MLM) 预训练目标来缓解前面提到的单向性约束。 掩蔽语言模型从输入中随机掩蔽一些标记，目标是仅根据其上下文来预测被掩蔽词的原始词汇表 id。 与从左到右的语言模型预训练不同，MLM 目标使表示能够融合左右上下文，这使我们能够预训练一个深度双向 Transformer。 除了掩码语言模型之外，我们还使用了“下一句预测”任务[给定两个句子，问这两个句子在原文中是否相邻]，该任务联合预训练文本对表示。 我们论文的贡献如下：：

- We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pre-trained deep bidirectional representations. This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs.

- 我们展示了双向预训练对语言表示的重要性。 与 Radford 等人不同。 (2018) 使用单向语言模型进行预训练，BERT 使用掩码语言模型来实现预训练的深度双向表示。 这也与彼得斯等人相反。 （2018a），它使用独立训练的从左到右和从右到左 LM 的浅连接。

- We show that pre-trained representations reduce the need for many heavily-engineered task-specific architectures. BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.

- 我们表明，预训练的表示减少了对许多精心设计的任务特定架构的需求。 BERT 是第一个基于微调的表示模型，它在大量句子级和词元级任务上实现了最先进的性能，优于许多特定于任务的架构。

- BERT advances the state of the art for eleven NLP tasks. The code and pre-trained models are available at https://github.com/google-research/bert.

- BERT 提升了 11 个 NLP 任务的最新技术水平。 代码和预训练模型可在 https://github.com/google-research/bert 获得。

## 2 Related Work 相关工作

There is a long history of pre-training general language representations, and we briefly review the most widely-used approaches in this section.

预训练通用语言表示的历史由来已久，我们将在本节简要回顾最广泛使用的方法。

### 2.1 Unsupervised Feature-based Approaches 无监督的基于特征的方法

Learning widely applicable representations of words has been an active area of research for decades, including non-neural (Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006) and neural (Mikolov et al., 2013; Pennington et al., 2014) methods. Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch (Turian et al., 2010). To pretrain word embedding vectors, left-to-right language modeling objectives have been used (Mnih and Hinton, 2009), as well as objectives to discriminate correct from incorrect words in left and right context (Mikolov et al., 2013).

几十年来，学习广泛适用的单词表示一直是一个活跃的研究领域，包括非神经（Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006）和神经（Mikolov et al., 2013） ; Pennington et al., 2014) 方法。 预训练的词嵌入是现代 NLP 系统不可或缺的一部分，与从头开始学习的嵌入相比，提供了显着的改进（Turian 等人，2010）。 为了预训练词嵌入向量，使用了从左到右的语言建模目标（Mnih 和 Hinton，2009），以及在左右上下文中区分正确单词和不正确单词的目标（Mikolov 等，2013）。

These approaches have been generalized to coarser granularities, such as sentence embeddings (Kiros et al., 2015; Logeswaran and Lee, 2018) or paragraph embeddings (Le and Mikolov, 2014). To train sentence representations, prior work has used objectives to rank candidate next sentences (Jernite et al., 2017; Logeswaran and Lee, 2018), left-to-right generation of next sentence words given a representation of the previous sentence (Kiros et al., 2015), or denoising autoencoder derived objectives (Hill et al., 2016).

这些方法已被推广到更粗粒度的方法，例如句子嵌入（Kiros et al., 2015; Logeswaran and Lee, 2018）或段落嵌入（Le and Mikolov, 2014）。 为了训练句子表示，先前的工作使用目标来对候选下一个句子进行排名（Jernite 等人，2017；Logeswaran 和 Lee，2018），给定前一个句子的表示，从左到右生成下一个句子单词（Kiros 等人） al., 2015)，或去噪自编码器派生目标 (Hill et al., 2016)。

ELMo and its predecessor (Peters et al., 2017, 2018a) generalize traditional word embedding research along a different dimension. They extract context-sensitive features from a left-to-right and a right-to-left language model. The contextual representation of each token is the concatenation of the left-to-right and right-to-left representations. When integrating contextual word embeddings with existing task-specific architectures, ELMo advances the state of the art for several major NLP benchmarks (Peters et al., 2018a) including question answering (Rajpurkar et al., 2016), sentiment analysis (Socher et al., 2013), and named entity recognition (Tjong Kim Sang and De Meulder, 2003). Melamud et al. (2016) proposed learning contextual representations through a task to predict a single word from both left and right context using LSTMs. Similar to ELMo, their model is feature-based and not deeply bidirectional. Fedus et al. (2018) shows that the cloze task can be used to improve the robustness of text generation models.

ELMo 及其前身 (Peters et al., 2017, 2018a) 将传统的词嵌入研究沿不同的维度进行了推广。他们从从左到右和从右到左的语言模型中提取上下文相关的特征。每个标记的上下文表示是从左到右和从右到左表示的连接。在将上下文词嵌入与现有的特定任务架构集成时，ELMo 提升了几个主要 NLP 基准（Peters 等人，2018a）的最新技术水平，包括问答（Rajpurkar 等人，2016）、情感分析（Socher 等人） ., 2013) 和命名实体识别 (Tjong Kim Sang and De Meulder, 2003)。梅拉姆德等人。 (2016) 提出通过使用 LSTM 从左右上下文预测单个单词的任务来学习上下文表示。与 ELMo 类似，他们的模型是基于特征的，而不是深度双向的。费杜斯等人。 (2018) 表明完形填空任务可用于提高文本生成模型的鲁棒性。

### 2.2 Unsupervised Fine-tuning Approaches 无监督微调方法

As with the feature-based approaches, the first works in this direction only pre-trained word embedding parameters from unlabeled text (Colobert and Weston, 2008).

与基于特征的方法一样，第一种方法在这个方向上只适用于未标记文本的预训练词嵌入参数（Colobert 和 Weston，2008）。

More recently, sentence or document encoders which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018). The advantage of these approaches is that few parameters need to be learned from scratch. At least partly due to this advantage, OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentence-level tasks from the GLUE benchmark (Wang et al., 2018a). Left-to-right language modeling and auto-encoder objectives have been used for pre-training such models (Howard and Ruder, 2018; Radford et al., 2018; Dai and Le, 2015).

最近，产生上下文标记表示的句子或文档编码器已经从未标记的文本中进行了预训练，并针对有监督的下游任务进行了微调（Dai 和 Le，2015；Howard 和 Ruder，2018；Radford 等人，2018）。 这些方法的优点是需要从头开始学习很少的参数。 至少部分由于这一优势，OpenAI GPT (Radford et al., 2018) 在 GLUE 基准 (Wang et al., 2018a) 的许多句子级任务上取得了先前最先进的结果。 从左到右的语言建模和自动编码器目标已用于预训练此类模型（Howard 和 Ruder，2018；Radford 等人，2018；Dai 和 Le，2015）。

### 2.3 Transfer Learning from Supervised Data 从监督数据迁移学习

There has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference (Conneau et al., 2017) and machine translation (McCann et al., 2017). Computer vision research has also demonstrated the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained with ImageNet (Deng et al., 2009; Yosinski et al., 2014).

还有一些工作显示了从具有大型数据集的监督任务中的有效迁移，例如自然语言推理 (Conneau et al., 2017) 和机器翻译 (McCann et al., 2017)。 计算机视觉研究还证明了从大型预训练模型进行迁移学习的重要性，其中一个有效的方法是微调使用 ImageNet 预训练的模型（Deng 等人，2009；Yosinski 等人，2014）。

## 3 BERT

We introduce BERT and its detailed implementation in this section. There are two steps in our framework: *pre-training and fine-tuning*. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section.

我们在本节介绍 BERT 及其详细实现。 我们的框架有两个步骤：*预训练和微调*。 在预训练期间，该模型通过不同的预训练任务在未标记数据上进行训练。 对于微调，BERT 模型首先使用预训练的参数进行初始化，然后使用来自下游任务的标记数据对所有参数进行微调。 每个下游任务都有单独的微调模型，即使它们使用相同的预训练参数进行初始化。 图 1 中的问答示例将作为本节的运行示例。

<div align= center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Fig%201.png"/></div>

Figure 1: Overall pre-training and fine-tuning procedures for BERT. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. During fine-tuning, all parameters are fine-tuned. `[CLS]` is a special symbol added in front of every input example, and `[SEP]` is a special separator token (e.g. separating questions/answers).
图 1：BERT 的整体预训练和微调程序。 除了输出层外，预训练和微调都使用相同的架构。 相同的预训练模型参数用于为不同的下游任务初始化模型。 在微调期间，所有参数都被微调。 `[CLS]` 是添加在每个输入示例前面的特殊符号，而 `[SEP]` 是特殊的分隔符（例如分隔问题/答案）。

A distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.

BERT 的一个显着特点是其跨不同任务的统一架构。 预训练的架构和最终的下游架构之间的差异很小。

**Model Architecture** BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the `tensor2tensor` library$^1$.  Because the use of Transformers has become common and our implementation is almost identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as “The Annotated Transformer.”$^2$

**模型架构** BERT 的模型架构是基于 Vaswani 等人描述的原始实现的多层双向 Transformer 编码器。 (2017) 并在 `tensor2tensor` 库$^1$ 中发布。 由于 Transformer 的使用已经变得普遍，而且我们的实现几乎与原来的相同，我们将省略模型架构的详尽背景描述，并请读者参考 Vaswani 等人。 （2017 年）以及“带注释的变形金刚”等优秀指南。$^2$

>$^1$https://github.com/tensorflow/tensor2tensor
>$^2$http://nlp.seas.harvard.edu/2018/04/03/attention.html

In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A$^3$.  We primarily report results on two model sizes: $\mathbf{ BERT_{BASE}}$ (${\rm L=12, H=768, A=12, Total Parameters=110M}$) and $\mathbf{ BERT_{LARGE}}$ (${\rm L=24, H=1024, A=16, Total Parameters=340M}$).

在这项工作中，我们将层数（即 Transformer 块）表示为 L，隐藏大小表示为 H，自注意力头数表示为 A$^3$。 我们主要报告两种模型大小的结果：$\mathbf{ BERT_{BASE}}$ (${\rm L=12, H=768, A=12, Total Parameters=110M}$) 和 $\mathbf{ BERT_{ 大}}$（${\rm L=24，H=1024，A=16，总参数=340M}$）。

>$^3$ In all cases we set the feed-forward/filter size to be $4H,
i.e., 3072 $for the $H = 768$ and $4096$ for the $H = 1024$.

$\mathbf{ BERT_{BASE}}$ was chosen to have the same model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.$^4$

$\mathbf{ BERT_{BASE}}$ 被选择为具有与 OpenAI GPT 相同的模型大小以进行比较。 然而，至关重要的是，BERT Transformer 使用双向自我注意，而 GPT Transformer 使用受限自我注意，其中每个标记只能关注其左侧的上下文。$^4$

>$^4$ 4We note that in the literature the bidirectional Transformer is often referred to as a “Transformer encoder” while the left-context-only version is referred to as a “Transformer decoder” since it can be used for text generation

**Input/Output Representations** To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., <Question, Answer>) in one token sequence. Throughout this work, a “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

**输入/输出表示**为了让 BERT 处理各种下游任务，我们的输入表示能够在一个标记序列中明确表示单个句子和一对句子（例如，<Question, Answer>） . 在整个工作中，“句子”可以是连续文本的任意跨度，而不是实际的语言句子。 一个“序列”是指输入到 BERT 的标记序列，它可能是一个句子，也可能是两个打包在一起的句子。

We use WordPiece embeddings (Wu et al., 2016) with a $30,000$ token vocabulary. The first token of every sequence is always a special classification token (`[CLS]`). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token (`[SEP]`). Second, we add a learned embedding to every token indicating whether it belongs to sentence `A` or sentence `B`. As shown in Figure 1, we denote input embedding as $E$, the final hidden vector of the special `[CLS]` token as $C \isin \mathbb{R}^H$, and the final hidden vector for the $i^{\rm th}$ input token as $T_i \isin \mathbb{R}^H$.

我们使用 WordPiece 嵌入（Wu 等人，2016 年） 30,000 个词汇词元。 每个序列的第一个标记始终是一个特殊的分类标记（`[CLS]`）。 与该标记对应的最终隐藏状态用作分类任务的聚合序列表示。 句子对被打包成一个单一的序列。 我们以两种方式区分句子。 首先，我们用一个特殊的标记（`[SEP]`）将它们分开。 其次，我们向每个标记添加一个学习嵌入，指示它属于句子“A”还是句子“B”。 如图 1 所示，我们将输入嵌入表示为 $E$，特殊“[CLS]”标记的最终隐藏向量表示为 $C \isin \mathbb{R}^H$，以及 $ i^{\rm th}$ 输入标记为 $T_i \isin \mathbb{R}^H$ 。

For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings. A visualization of this construction can be seen in Figure 2.

对于给定的词元，其输入表示是通过对相应的词元、第几个句子和在序列中的位置嵌入求和来构造的。 这种结构的可视化可以在图 2 中看到。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Fig%202.png"/></div>

Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.
图 2：BERT 输入表示。 输入嵌入是词元嵌入、分割嵌入和位置嵌入的总和。

### 3.1 Pre-training BERT 预训练BERT

Unlike Peters et al. (2018a) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step is presented in the left part of Figure 1.

不像彼得斯等人。 (2018a) 和 Radford 等人。 (2018)，我们不使用传统的从左到右或从右到左的语言模型来预训练 BERT。 相反，我们使用本节中描述的两个无监督任务来预训练 BERT。 此步骤显示在图 1 的左侧。

**Task #1: Masked LM** Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.

**任务#1: Masked LM**。从直觉上讲，我们有理由相信，深度双向模型严格来说比从左到右的模型或从左到右和从右到左的浅层连接模型更强大。不幸的是，标准的条件语言模型只能从左到右或从右到左进行训练，因为双向条件将允许每个词间接地 "看到自己"，并且该模型可以在多层次的背景下简单地预测目标词。

In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a *Cloze* task in the literature (Taylor, 1953). In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask $15\%$ of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.

为了训练深度双向表示，我们简单地随机屏蔽一定百分比的输入标记，然后预测那些被屏蔽的标记。 我们将此过程称为“掩码 LM”（MLM），尽管它在文献中通常被称为 *Cloze* 任务（Taylor，1953）。 在这种情况下，与掩码标记对应的最终隐藏向量被馈送到词汇表上的输出 softmax，就像在标准 LM 中一样。 在我们所有的实验中，我们随机屏蔽了每个序列中所有 WordPiece 标记的 $15\%$。 与去噪自动编码器（Vincent et al., 2008）相比，我们只预测被掩蔽的单词而不是重建整个输入。

Although this allows us to obtain a bidirectional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the `[MASK]` token does not appear during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual `[MASK]` token. The training data generator chooses 15% of the token positions at random for prediction. If the $i$-th token is chosen, we replace the i-th token with (1) the `[MASK]` token $80\%$ of the time (2) a random token $10\%$ of the time (3) the unchanged $i$-th token $10\%$ of the time. Then, $T_i$ will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.

尽管这使我们能够获得双向预训练模型，但缺点是我们在预训练和微调之间造成了不匹配，因为在微调期间不会出现“[MASK]”标记。 为了缓解这种情况，我们并不总是用实际的 `[MASK]` 标记替换“掩码”单词。 训练数据生成器随机选择 15% 的标记位置进行预测。 如果选择了第 i 个词元，我们将第 i 个词元以替换为 (1) $80\%$概率为`[MASK]` 词元 (2)$10\%$概率为随机词元  (3) $10\%$概率不变的该$i$词元。 然后，$T_i$ 将用于预测具有交叉熵损失的原始词元。 我们在附录 C.2 中比较了此过程的变体。

**Task #2: Next Sentence Prediction (NSP)** Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the *relationship* between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized *next sentence prediction* task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences `A` and `B` for each pre-training example, $50\%$ of the time `B` is the actual next sentence that follows `A` (labeled as `IsNext`), and $50\%$ of the time it is a random sentence from the corpus (labeled as `NotNext`). As we show in Figure 1, $C$ is used for next sentence prediction (NSP)$^5$.  Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI.$^6$

**任务2：下一句预测（NSP）**许多重要的下游任务，如问题回答（QA）和自然语言推理（NLI），都是基于对两个句子之间*关系*的理解，而语言建模并不能直接捕捉到这种关系。为了训练一个能够理解句子关系的模型，我们对一个二进制的*下一句预测*任务进行预训练，这个任务可以从任何单语语料库中简单地生成。具体来说，在为每个预训练例子选择句子`A`和`B`时，50％的概率`B`是紧随`A`的实际下一句（标记为`IsNext`），50％的概率是语料库中的一个随机句子（标记为`NotNext`）。如我们在图1中所示，C$被用于下一句预测（NSP）$^5$。 尽管它很简单，但我们在第5.1节中证明，针对这一任务的预训练对质量保证和无法律约束力的工作都是非常有益的$^6$。

>$^5$ The final model achieves $97\%-98\%$ accuracy on NSP
>$^6$ The vector $C$ is not a meaningful sentence representation without fine-tuning, since it was trained with NSP.

The NSP task is closely related to representation-learning objectives used in Jernite et al. (2017) and Logeswaran and Lee (2018). However, in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters.

NSP任务与Jernite等人（2017）和Logeswaran和Lee（2018）中使用的表示学习目标密切相关。然而，在之前的工作中，只有句子嵌入被转移到下游任务中，而BERT转移所有参数以初始化终端任务模型参数

**Pre-training** data The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.

**预训练**数据预训练过程很大程度上遵循现有的语言模型预训练文献。 对于预训练语料库，我们使用 BooksCorpus（8 亿字）（Zhu et al., 2015）和英语维基百科（25 亿字）。 对于 Wikipedia，我们只提取文本段落并忽略列表、表格和标题。 为了提取长的连续序列，使用文档级语料库而不是像十亿字基准（Chelba et al., 2013）这样的打乱句子级语料库至关重要。

### 3.2 Fine-tuning BERT 微调BERT

Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks——whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs. For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention, such as Parikh et al. (2016); Seo et al. (2017). BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes *bidirectional* cross attention between two sentences.

微调很简单，因为 Transformer 中的自注意力机制允许 BERT 通过交换适当的输入和输出来对许多下游任务（无论它们涉及单个文本还是文本对）进行建模。 对于涉及文本对的应用程序，一种常见的模式是在应用双向交叉注意之前独立编码文本对，例如 Parikh 等人。 （2016）； 徐等人。 （2017）。 BERT 使用自注意力机制来统一这两个阶段，因为使用自注意力对连接的文本对进行编码有效地包括了两个句子之间的*双向*交叉注意力。

For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end. At the input, sentence `A` and sentence `B` from pre-training are analogous to (1) sentence pairs in paraphrasing, (2) hypothesis-premise pairs in entailment, (3) question-passage pairs in question answering, and (4) a degenerate text-$\emptyset$ pair in text classification or sequence tagging. At the output, the token representations are fed into an output layer for token-level tasks, such as sequence tagging or question answering, and the `[CLS]` representation is fed into an output layer for classification, such as entailment or sentiment analysis.

对于每个任务，我们只需将特定于任务的输入和输出插入 BERT 并端到端微调所有参数。 在输入端，来自预训练的句子“A”和句子“B”类似于（1）释义中的句子对，（2）蕴涵中的假设-前提对，（3）问答中的问题-段落对，以及 (4) 文本分类或序列标注中的退化文本-$\emptyset$ 对。 在输出端，token 表示被输入到输出层用于标记级任务，例如序列标记或问答，而“[CLS]”表示被输入到输出层进行分类，例如蕴含或情感分析 .

Compared to pre-training, fine-tuning is relatively inexpensive. All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model.$^7$ We describe the task-specific details in the corresponding subsections of Section 4. More details can be found in Appendix A.5.

与预训练相比，微调相对便宜。 论文中的所有结果最多可以在单个 Cloud TPU 上运行 1 小时，或者在 GPU 上运行几个小时，从完全相同的预训练模型开始。$^7$ 我们描述了特定于任务的细节 在第 4 节的相应小节中。更多细节可以在附录 A.5 中找到。

>$^7$ For example, the BERT SQuAD model can be trained in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of $91.0\%$

## 4 Experiments

In this section, we present BERT fine-tuning results on 11 NLP tasks.

### 4.1 GLUE

The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018a) is a collection of diverse natural language understanding tasks. Detailed descriptions of GLUE datasets are included in Appendix B.1.

To fine-tune on GLUE, we represent the input sequence (for single sentence or sentence pairs) as described in Section 3, and use the final hidden vector $C \isin \mathbb{R}^H$ corresponding to the first input token (`[CLS]`) as the aggregate representation. The only new parameters introduced during fine-tuning are classification layer weights $W \isin \mathbb{R}^{K \times H}$ , where $K$ is the number of labels. We compute a standard classification loss with $C$ and $W$, i.e., $\log({\rm softmax}(CW^T))$.

为了在 GLUE 上进行微调，我们表示输入序列（对于单个句子或句子对），如第 3 节所述，并使用与第一个输入标记对应的最终隐藏向量 $C \isin \mathbb{R}^H$ (`[CLS]`) 作为聚合表示。 在微调期间引入的唯一新参数是分类层权重 $W \isin \mathbb{R}^{K \times H}$ ，其中 $K$ 是标签的数量。 我们用 $C$ 和 $W$ 计算标准分类损失，即 $\log({\rm softmax}(CW^T))$。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%201.png"/></div>

Table 1: GLUE Test results, scored by the evaluation server (https://gluebenchmark.com/leaderboard).The number below each task denotes the number of training examples. The “Average” column is slightly different than the official GLUE score, since we exclude the problematic WNLI set.$^8$ BERT and OpenAI GPT are single-model, single task. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks. We exclude entries that use BERT as one of their components.

>$^8$See (10) in https://gluebenchmark.com/faq

We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set. Additionally, for $\mathbf{BERT_{LARGE}}$ we found that finetuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set. With random restarts, we use the same pre-trained checkpoint but perform different fine-tuning data shuffling and classifier layer initialization$^9$.

>$^9$ The GLUE data set distribution does not include the Test labels, and we only made a single GLUE evaluation server submission for each of $\mathbf{BERT_{BASE}}$ and $\mathbf {BERT_{LARGE}}$.

Results are presented in Table 1. Both $\mathbf{BERT_{BASE}}$ and $\mathbf{BERT_{LARGE}}$ outperform all systems on all tasks by a substantial margin, obtaining $4.5\%$ and $7.0\%$ respective average accuracy improvement over the prior state of the art. Note that $\mathbf{BERT_{BASE}}$ and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking. For the largest and most widely reported GLUE task, MNLI, BERT obtains a $4.6\%$ absolute accuracy improvement. On the official GLUE leaderboard$^{10}$, $\mathbf{BERT_{LARGE}}$ obtains a score of $80.5$, compared to OpenAI GPT, which obtains $72.8$ as of the date of writing.

>$^{10}$ https://gluebenchmark.com/leaderboard

We find that $\mathbf {BERT_{LARGE}}$ significantly outperforms $\mathbf{BERT_{BASE}}$ across all tasks, especially those with very little training data. The effect of model size is explored more thoroughly in Section 5.2.

### 4.2 SQuAD v1.1

The Stanford Question Answering Dataset (SQuAD v1.1) is a collection of 100k crowdsourced question/answer pairs (Rajpurkar et al., 2016). Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage.

As shown in Figure 1, in the question answering task, we represent the input question and passage as a single packed sequence, with the question using the A embedding and the passage using the `B` embedding. We only introduce a start vector $S \isin \mathbb{R}^H$ and an end vector $E \isin \mathbb{R}^H$ during fine-tuning. The probability of word $i$ being the start of the answer span is computed as a dot prod-uct between $T_i$ and $S$ followed by a softmax over eSTi all of the words in the paragraph: $P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}} $. The analogous formula is used for the end of the answer span. The score of a candidate span from position $i$ to position $j$ is defined as $S \cdot T_i + E \cdot T_j$ , and the maximum scoring span where $j  \geq  i$ is used as a prediction. The training objective is the sum of the log-likelihoods of the correct start and end positions. We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32.

Table 2 shows top leaderboard entries as well as results from top published systems (Seo et al., 2017; Clark and Gardner, 2018; Peters et al., 2018a; Hu et al., 2018). The top results from the SQuAD leaderboard do not have up-to-date public system descriptions available,$^{11}$ and are allowed to use any public data when training their systems. We therefore use modest data augmentation in our system by first fine-tuning on TriviaQA (Joshi et al., 2017) befor fine-tuning on SQuAD.

>$^{11}$QANet is described in Yu et al. (2018), but the system has improved substantially after publication

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%202.png"/></div>

Table 2: SQuAD 1.1 results. The BERT ensemble is 7x systems which use different pre-training checkpoints and fine-tuning seeds.

Our best performing system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. In fact, our single BERT model outperforms the top ensemble system in terms of F1 score. Without TriviaQA fine-tuning data, we only lose 0.1-0.4 F1, still outper-forming all existing systems by a wide margin$^{12}$.

>$^{12}$ The TriviaQA data we used consists of paragraphs from TriviaQA-Wiki formed of the first 400 tokens in documents,that contain at least one of the provided possible answers.

### 4.3 SQuAD v2.0

The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.

We use a simple approach to extend the SQuAD v1.1 BERT model for this task. We treat questions that do not have an answer as having an answer span with start and end at the `[CLS]` token. The probability space for the start and end answer span positions is extended to include the position of the `[CLS]` token. For prediction, we compare the score of the no-answer span: $s_{null} = S \cdot C + E \cdot C$ to the score of the best non-null span $\hat{s_{i,j}} = \max_{j \geq i} S \cdot T_i + E \cdot T_j$. We predict a non-null answer when $\hat{s_{i,j}} > s_{null} + \tau$, where the threshold is selected on the dev set to maximize F1. We did not use TriviaQA data for this model. We fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.

> Bert 微调时训练epoch应该要大，此外，Bert使用的Adam优化器不是正常的Adam优化器，使用时应该要换成正常的Adam优化器

The results compared to prior leaderboard entries and top published work (Sun et al., 2018; Wang et al., 2018b) are shown in Table 3, excluding systems that use BERT as one of their components. We observe a +5.1 F1 improvement over the previous best system.

<div align= center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%203.png"/></div>

Table 3: SQuAD 2.0 results. We exclude entries that use BERT as one of their components

### 4.4 SWAG

The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference (Zellers et al., 2018). Given a sentence, the task is to choose the most plausible continuation among four choices.

When fine-tuning on the SWAG dataset, we construct four input sequences, each containing the concatenation of the given sentence (sentence `A`) and a possible continuation (sentence `B`). The only task-specific parameters introduced is a vector whose dot product with the `[CLS]` token representation $C$ denotes a score for each choice which is normalized with a softmax layer.

We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16. Results are presented in Table 4. $\mathbf{BERT_{LARGE}}$ outperforms the authors’ baseline ESIM+ELMo system by $+27.1\%$ and OpenAI GPT by $8.3\%$.

<div align= center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%204.png"/></div>

Table 4: SWAG Dev and Test accuracies. †Human performance is measured with 100 samples, as reported in the SWAG paper.

## 5 Ablation Studies 

In this section, we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance. Additional ablation studies can be found in Appendix C.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%205.png"/></div>

Table 5: Ablation over the pre-training tasks using the BERTBASE architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a randomly initialized BiLSTM on top of the “LTR + No NSP” model during fine-tuning.

### 5.1 Effect of Pre-training Tasks

We demonstrate the importance of the deep bidi-rectionality of BERT by evaluating two pretraining objectives using exactly the same pretraining data, fine-tuning scheme, and hyperparameters as BERTBASE :

**No NSP**: A bidirectional model which is trained using the “masked LM” (MLM) but without the “next sentence prediction” (NSP) task.

**LTR & No NSP**: A left-context-only model which is trained using a standard Left-to-Right (LTR) LM, rather than an MLM. The left-only constraint was also applied at fine-tuning, because removing it introduced a pre-train/fine-tune mismatch that degraded downstream performance. Additionally, this model was pre-trained without the NSP task. This is directly comparable to OpenAI GPT, but using our larger training dataset, our input representation, and our fine-tuning scheme.

We first examine the impact brought by the NSP task. In Table 5, we show that removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1. Next, we evaluate the impact of training bidirectional representations by comparing “No NSP” to “LTR & No NSP”. The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD.

For SQuAD it is intuitively clear that a LTR model will perform poorly at token predictions, since the token-level hidden states have no rightside context. In order to make a good faith attempt at strengthening the LTR system, we added a randomly initialized BiLSTM on top. This does significantly improve results on SQuAD, but the results are still far worse than those of the pretrained bidirectional models. The BiLSTM hurts performance on the GLUE tasks.

We recognize that it would also be possible to train separate LTR and RTL models and represent each token as the concatenation of the two models, as ELMo does. However: (a) this is twice as expensive as a single bidirectional model; (b) this is non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question; (c) this it is strictly less powerful than a deep bidirectional model, since it can use both left and right context at every layer.

### 5.2 Effect of Model Size 模型大小的影响

In this section, we explore the effect of model size on fine-tuning task accuracy. We trained a number of BERT models with a differing number of layers, hidden units, and attention heads, while otherwise using the same hyperparameters and training procedure as described previously.

Results on selected GLUE tasks are shown in Table 6. In this table, we report the average Dev Set accuracy from 5 random restarts of fine-tuning. We can see that larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples, and is substantially different from the pre-training tasks. It is also perhaps surprising that we are able to achieve such significant improvements on top of models which are already quite large relative to the existing literature. For example, the largest Transformer explored in Vaswani et al. (2017) is $\rm (L=6, H=1024, A=16)$ with 100M parameters for the encoder, and the largest Transformer we have found in the literature is $\rm (L=64, H=512, A=2)$ with 235M parameters (Al-Rfou et al., 2018). By contrast, BERTBASE contains 110M parameters and BERTLARGE contains 340M parameters.

It has long been known that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling, which is demonstrated by the LM perplexity of held-out training data shown in Table 6. However, we believe that this is the first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained. Peters et al. (2018b) presented mixed results on the downstream task impact of increasing the pre-trained bi-LM size from two to four layers and Melamud et al. (2016) mentioned in passing that increasing hidden dimension size from 200 to 600 helped, but increasing further to 1,000 did not bring further improvements. Both of these prior works used a featurebased approach — we hypothesize that when the model is fine-tuned directly on the downstream tasks and uses only a very small number of randomly initialized additional parameters, the taskspecific models can benefit from the larger, more expressive pre-trained representations even when downstream task data is very small.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%206.png"/></div>

Table 6: Ablation over BERT model size. #L = the number of layers; #H = hidden size; #A = number of attention heads. “LM (ppl)” is the masked LM perplexity of held-out training data.

### 5.3 Feature-based Approach with BERT

All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added to the pre-trained model, and all parameters are jointly fine-tuned on a downstream task. However, the feature-based approach, where fixed features are extracted from the pretrained model, has certain advantages. First, not all tasks can be easily represented by a Transformer encoder architecture, and therefore require a task-specific model architecture to be added. Second, there are major computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top of this representation.

In this section, we compare the two approaches by applying BERT to the CoNLL-2003 Named Entity Recognition (NER) task (Tjong Kim Sang and De Meulder, 2003). In the input to BERT, we use a case-preserving WordPiece model, and we include the maximal document context provided by the data. Following standard practice, we formulate this as a tagging task but do not use a CRF layer in the output. We use the representation of the first sub-token as the input to the token-level classifier over the NER label set.

To ablate the fine-tuning approach, we apply the feature-based approach by extracting the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiLSTM before the classification layer.

Results are presented in Table 7. BERTLARGE performs competitively with state-of-the-art methods. The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer, which is only 0.3 F1 behind fine-tuning the entire model. This demonstrates that BERT is effective for both finetuning and feature-based approaches.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%207.png"/></div>

Table 7: CoNLL-2003 Named Entity Recognition results. Hyperparameters were selected using the Dev set. The reported Dev and Test scores are averaged over 5 random restarts using those hyperparameters.

## 6 Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures. Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.

最近由于使用语言模型进行迁移学习的经验改进表明，丰富的、无监督的预训练是许多语言理解系统不可或缺的一部分。 特别是，这些结果使即使是低资源任务也能从深度单向架构中受益。 我们的主要贡献是将这些发现进一步推广到深度双向架构，允许相同的预训练模型成功处理广泛的 NLP 任务。

## References

Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string embeddings for sequence labeling. In Proceedings of the 27th International Conference on Computational Linguistics, pages 1638-1649.

Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. 2018. Character-level language modeling with deeper self-attention. arXiv preprint arXiv:1808.04444.

Rie Kubota Ando and Tong Zhang. 2005. A framework for learning predictive structures from multiple tasks and unlabeled data. Journal of Machine Learning Research, 6(Nov):1817-1853.

Luisa Bentivogli, Bernardo Magnini, Ido Dagan, Hoa Trang Dang, and Danilo Giampiccolo. 2009. The fifth PASCAL recognizing textual entailment challenge. In TAC. NIST.

John Blitzer, Ryan McDonald, and Fernando Pereira. 2006. Domain adaptation with structural correspon-dence learning. In Proceedings of the 2006 conference on empirical methods in natural language processing, pages 120-128. Association for Computational Linguistics.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In EMNLP. Association for Computational Linguistics.

Peter F Brown, Peter V Desouza, Robert L Mercer, Vincent J Della Pietra, and Jenifer C Lai. 1992. Class-based n-gram models of natural language. Computational linguistics, 18(4):467-479.

Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez- Gazpio, and Lucia Specia. 2017. Semeval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 1-14, Vancouver, Canada. Association for Computational Linguistics.

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2013. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005.

Z. Chen, H. Zhang, X. Zhang, and L. Zhao. 2018. Quora question pairs.

Christopher Clark and Matt Gardner. 2018. Simple and effective multi-paragraph reading comprehension. In ACL.

Kevin Clark, Minh-Thang Luong, Christopher D Man-ning, and Quoc Le. 2018. Semi-supervised sequence modeling with cross-view training. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 19141925.

Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Pro-ceedings of the 25th international conference on Machine learning, pages 160-167. ACM.

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. 2017. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670-680, Copenhagen, Denmark. Association for Computational Linguistics.

Andrew M Dai and Quoc V Le. 2015. Semi-supervised sequence learning. In Advances in neural information processing systems, pages 3079-3087.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei- Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09.

William B Dolan and Chris Brockett. 2005. Automati-cally constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005).

William Fedus, Ian Goodfellow, and Andrew M Dai. 2018. Maskgan: Better text generation via filling in the_. arXivpreprint arXiv:180L07736.

Dan Hendrycks and Kevin Gimpel. 2016. Bridging nonlinearities and stochastic regularizers with gaus-sian error linear units. CoRR, abs/1606.08415.

Felix Hill, Kyunghyun Cho, and Anna Korhonen. 2016. Learning distributed representations of sentences from unlabelled data. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computa-tional Linguistics.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In ACL. Association for Computational Linguistics.

Minghao Hu, Yuxing Peng, Zhen Huang, Xipeng Qiu, Furu Wei, and Ming Zhou. 2018. Reinforced mnemonic reader for machine reading comprehension. In IJCAI.

Yacine Jernite, Samuel R. Bowman, and David Son- tag. 2017. Discourse-based objectives for fast un-supervised sentence representation learning. CoRR, abs/1705.00557.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehen-sion. In ACL.

Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Skip-thought vectors. In Advances in neural information processing systems, pages 3294-3302.

Quoc Le and Tomas Mikolov. 2014. Distributed rep-resentations of sentences and documents. In Inter-national Conference on Machine Learning, pages 1188-1196.

Hector J Levesque, Ernest Davis, and Leora Morgenstern. 2011. The winograd schema challenge. In Aaai spring symposium: Logical formalizations of commonsense reasoning, volume 46, page 47.

Lajanugen Logeswaran and Honglak Lee. 2018. An efficient framework for learning sentence represen-tations. In International Conference on Learning Representations.

Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Con-textualized word vectors. In NIPS.

Oren Melamud, Jacob Goldberger, and Ido Dagan. 2016. context2vec: Learning generic context embedding with bidirectional LSTM. In CoNLL.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Cor-rado, and Jeff Dean. 2013. Distributed representa-tions of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26, pages 3111-3119. Curran Associates, Inc.

Andriy Mnih and Geoffrey E Hinton. 2009. A scalable hierarchical distributed language model. In D. Koller, D. Schuurmans, Y. Bengio, and L. Bot- tou, editors, Advances in Neural Information Processing Systems 21, pages 1081-1088. Curran Associates, Inc.

Ankur P Parikh, Oscar Tackstrom, Dipanjan Das, and Jakob Uszkoreit. 2016. A decomposable attention model for natural language inference. In EMNLP.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 15321543.

Matthew Peters, Waleed Ammar, Chandra Bhagavat- ula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In ACL.

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018a. Deep contextualized word rep-resentations. In NAACL.

Matthew Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. 2018b. Dissecting contextual word embeddings: Architecture and representation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1499-1509.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language under-standing with unsupervised learning. Technical report, OpenAI.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392.

Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. 2017. Bidirectional attention flow for machine comprehension. In ICLR.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631-1642.

Fu Sun, Linyang Li, Xipeng Qiu, and Yang Liu. 2018. U-net: Machine reading comprehension with unanswerable questions. arXiv preprint arXiv:1810.06638.

Wilson L Taylor. 1953. Cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30(4):415-433.

Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Language-independent named entity recognition. In CoNLL.

Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: A simple and general method for semi-supervised learning. In Proceedings of the 48th Annual Meeting of the Association for Compu-tational Linguistics, ACL ’10, pages 384-394.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Pro-cessing Systems, pages 6000-6010.

Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoen-coders. In Proceedings of the 25th international conference on Machine learning, pages 1096-1103. ACM.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018a. Glue: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353-355.

Wei Wang, Ming Yan, and Chen Wu. 2018b. Multi-granularity hierarchical attention fusion networks for reading comprehension and question answering. In Proceedings of the 56th Annual Meeting of the As-sociation for Computational Linguistics (Volume 1: Long Papers). Association for Computational Lin-guistics.

Alex Warstadt, Amanpreet Singh, and Samuel R Bow-man. 2018. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471.

Adina Williams, Nikita Nangia, and Samuel R Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In NAACL.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks? In Advances in neural information processing systems, pages 3320-3328.

Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V Le. 2018. QANet: Combining local convolution with global self-attention for reading comprehension. In ICLR.

Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhut- dinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages 19-27.

## Appendix for “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”

We organize the appendix into three sections:

- Additional implementation details for BERT are presented in Appendix A;
- Additional details for our experiments are presented in Appendix B; and
- Additional ablation studies are presented in Appendix C.

    We present additional ablation studies for BERT including:
    - Effect ofNumber ofTraining Steps; and
    - Ablation for Different Masking Procedures.

## A Additional Details for BERT

### A.1 Illustration of the Pre-training Tasks

We provide examples of the pre-training tasks in the following.

**Masked LM and the Masking Procedure** Assuming the unlabeled sentence is `my dog is hairy`, and during the random masking procedure we chose the 4-th token (which corresponding to `hairy`), our masking procedure can be further illustrated by

- 80% of the time: Replace the word with the `[MASK]` token, e.g.,
  > my dog is hairy  $\rightarrow$ my dog is `[MASK]`
- 10% of the time: Replace the word with a random word, e.g.,
  > my dog is hairy  $\rightarrow$ my dog is apple
- 10% of the time: Keep the word unchanged, e.g.,
  > my dog is hairy $\rightarrow$ my dog is hairy.

  The purpose of this is to bias the representation towards the actual observed word.这样做的目的是使表示偏向于实际观察到的单词

The advantage of this procedure is that the Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Additionally, because random replacement only occurs for 1.5% of all tokens (i.e., 10% of 15%), this does not seem to harm the model’s language understanding capability. In Section C.2, we evaluate the impact this procedure.

Compared to standard langauge model training, the masked LM only make predictions on 15% of tokens in each batch, which suggests that more pre-training steps may be required for the model to converge. In Section C.1 we demonstrate that MLM does converge marginally slower than a left- to-right model (which predicts every token), but the empirical improvements of the MLM model far outweigh the increased training cost.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Fig%203.png"/></div>

Figure 3: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-to- left LSTMs to generate features for downstream tasks. Among the three, only BERT representations are jointly conditioned on both left and right context in all layers. In addition to the architecture differences, BERT and OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.

**Next Sentence Prediction** The next sentence prediction task can be illustrated in the following examples.

Input = `[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]`

Label = `IsNext`

Input = `[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]`

Label = `NotNext`

### A.2 Pre-training Procedure

To generate each training input sequence, we sample two spans of text from the corpus, which we refer to as “sentences” even though they are typically much longer than single sentences (but can be shorter also). The first sentence receives the `A` embedding and the second receives the `B` embedding. 50% of the time `B` is the actual next sentence that follows `A` and 50% of the time it is a random sentence, which is done for the “next sentence prediction” task. They are sampled such that the combined length is $\le$ 512 tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%, and no special consideration given to partial word pieces.

We train with batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. We use Adam with learning rate of 1e-4, $\beta_1 =0.9$, $\beta_2 = 0.999$, L2 weight decay of 0:01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a `gelu` activation (Hendrycks and Gimpel, 2016) rather than the standard `relu`, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood.

Training of $\mathbf {BERT_{BASE}}$ was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total)$^{13}$.  Training of BERTLARGE was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete.

>$^{13}$ https://cloudplatform.googleblog.com/2018/06/Cloud-TPU-now-offers-preemptible-pricing-and-global-availability.html

Longer sequences are disproportionately expensive because attention is quadratic to the sequence length. To speed up pretraing in our experiments, we pre-train the model with sequence length of 128 for 90% of the steps. Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings.

### A.3 Fine-tuning Procedure

For fine-tuning, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The dropout probability was always kept at 0.1. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:

- **Batch size**: 16, 32
- **Learning rate (Adam):** 5e-5, 3e-5, 2e-5
- **Number of epoch**s: 2, 3, 4

We also observed that large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to simply run an exhaustive search over the above parameters and choose the model that performs best on the development set.

### A.4 Comparison of BERT, ELMo ,and OpenAI GPT

Here we studies the differences in recent popular representation learning models including ELMo, OpenAI GPT and BERT. The comparisons between the model architectures are shown visually in Figure 3. Note that in addition to the architecture differences, BERT and OpenAI GPT are finetuning approaches, while ELMo is a feature-based approach.

The most comparable existing pre-training method to BERT is OpenAI GPT, which trains a left-to-right Transformer LM on a large text corpus. In fact, many of the design decisions in BERT were intentionally made to make it as close to GPT as possible so that the two methods could be minimally compared. The core argument of this work is that the bi-directionality and the two pretraining tasks presented in Section 3.1 account for the majority of the empirical improvements, but we do note that there are several other differences between how BERT and GPT were trained:

- GPT is trained on the BooksCorpus (800M words); BERT is trained on the BooksCor- pus (800M words) and Wikipedia (2,500M words).
- GPT uses a sentence separator (`[SEP]`) and classifier token (`[CLS]`) which are only introduced at fine-tuning time; BERT learns `[SEP]`, `[CLS]` and sentence `A/B` embeddings during pre-training.
- GPT was trained for 1M steps with a batch size of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words.
- GPT used the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate which performs the best on the development set.

To isolate the effect of these differences, we perform ablation experiments in Section 5.1 which demonstrate that the majority of the improvements are in fact coming from the two pre-training tasks and the bidirectionality they enable.

### A.5 Illustrations of Fine-tuning on Different Tasks

The illustration of fine-tuning BERT on different tasks can be seen in Figure 4. Our task-specific models are formed by incorporating BERT with one additional output layer, so a minimal number of parameters need to be learned from scratch. Among the tasks, (a) and (b) are sequence-level tasks while (c) and (d) are token-level tasks. In the figure, E represents the input embedding, Ti represents the contextual representation of token i, `[CLS]` is the special symbol for classification output, and `[SEP]` is the special symbol to separate non-consecutive token sequences.

## B Detailed Experimental Setup

### B.1 Detailed Descriptions for the GLUE Benchmark Experiments.

Our GLUE results in Table1 are obtained from https://gluebenchmark.com/ leaderboard and https://blog.openai.com/language-unsupervised. The GLUE benchmark includes the following datasets, the descriptions of which were originally summarized in Wang et al. (2018a):

**MNLI** Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task (Williams et al., 2018). Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one.

**QQP** Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent (Chen et al., 2018).

**QNLI** Question Natural Language Inference is a version of the Stanford Question Answering Dataset (Rajpurkar et al., 2016) which has been converted to a binary classification task (Wang et al., 2018a). The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Fig%204.png"/></div>
 
Figure 4: Illustrations of Fine-tuning BERT on Different Tasks. 

**SST-2** The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment (Socher et al., 2013).

**CoLA** The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not (Warstadt et al., 2018).

**STS-B** The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources (Cer et al., 2017). They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning.

**MRPC** Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent (Dolan and Brockett, 2005).

**RTE** Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data (Bentivogli et al., 2009).$^{14}$

>$^{14}$ Note that we only report single-task fine-tuning results in this paper. A multitask fine-tuning approach could potentially push the performance even further. For example, we did observe substantial improvements on RTE from multitask training with MNLI.

**WNLI** Winograd NLI is a small natural language inference dataset (Levesque et al., 2011). The GLUE webpage notes that there are issues with the construction of this dataset,$^{15}$ and every trained system that’s been submitted to GLUE has performed worse than the 65.1 baseline accuracy of predicting the majority class. We therefore exclude this set to be fair to OpenAI GPT. For our GLUE submission, we always predicted the majority class.

>$^{15}$ https://gluebenchmark.com/faq

## C Additional Ablation Studies

### C.1 Effect of Number of Training Steps

Figure 5 presents MNLI Dev accuracy after finetuning from a checkpoint that has been pre-trained for k steps. This allows us to answer the following questions:

1. Question: Does BERT really need such a large amount of pre-training (128,000 words/batch * 1,000,000 steps) to achieve high fine-tuning accuracy?
Answer: Yes, BERTBASE achieves almost 1.0% additional accuracy on MNLI when trained on 1M steps compared to 500k steps.

2. Question: Does MLM pre-training converge slower than LTR pre-training, since only 15% of words are predicted in each batch rather than every word?
Answer: The MLM model does converge slightly slower than the LTR model. However, in terms of absolute accuracy the MLM model begins to outperform the LTR model almost immediately.

### C.2 Ablation for Different Masking Procedures

In Section 3.1, we mention that BERT uses a mixed strategy for masking the target tokens when pre-training with the masked language model (MLM) objective. The following is an ablation study to evaluate the effect of different masking strategies.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Fig%205.png"/></div>
 
Figure 5: Ablation over number of training steps. This shows the MNLI accuracy after fine-tuning, starting from model parameters that have been pre-trained for $k$ steps. The x-axis is the value of $k$.

Note that the purpose of the masking strategies is to reduce the mismatch between pre-training and fine-tuning, as the `[MASK]` symbol never appears during the fine-tuning stage. We report the Dev results for both MNLI and NER. For NER, we report both fine-tuning and feature-based approaches, as we expect the mismatch will be amplified for the feature-based approach as the model will not have the chance to adjust the representations.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding/Table%208.png"/></div>

Table 8: Ablation over different masking strategies.

The results are presented in Table 8. In the table, MASK means that we replace the target token with the `[MASK]` symbol for MLM; SAME means that we keep the target token as is; RND means that we replace the target token with another random token.

The numbers in the left part of the table represent the probabilities of the specific strategies used during MLM pre-training (BERT uses $80\%, 10\%, 10\%$). The right part of the paper represents the Dev set results. For the feature-based approach, we concatenate the last 4 layers of BERT as the features, which was shown to be the best approach in Section 5.3.

From the table it can be seen that fine-tuning is surprisingly robust to different masking strategies. However, as expected, using only the MASK strategy was problematic when applying the featurebased approach to NER. Interestingly, using only the RND strategy performs much worse than our strategy as well.
