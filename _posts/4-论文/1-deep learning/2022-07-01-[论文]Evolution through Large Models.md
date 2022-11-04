---
title: 【深度学习】通过大型模型的演化 Evolution through Large Models
date: 2022-07-15 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
---

## Abstract 摘要

This paper pursues the insight that large language models (LLMs) trained to generate code can vastly improve the effectiveness of mutation operators applied to programs in genetic programming (GP). Because such LLMs benefit from training data that includes sequential changes and modifications, they can approximate likely changes that humans would make. To highlight the breadth of implications of such evolution through large models (ELM), in the main experiment ELM combined with MAPElites generates hundreds of thousands of functional examples of Python programs that output working ambulating robots in the Sodarace domain,which the original LLM had never seen in pre-training. These examples then help to bootstrap training a new conditional language model that can output the right walker for a particular terrain. The ability to bootstrap new models that can output appropriate artifacts for a given context in a domain where zero training data was previously available carries implications for open-endedness, deep learning, and reinforcement learning. These implications are explored here in depth in the hope of inspiring newdirections of research now opened up by ELM.

本文探讨了训练生成代码的大型语言模型（LLM）可以极大地提高应用于遗传编程（GP）程序的变异算子的有效性。由于此类LLM受益于包含连续变化和修改的训练数据，因此它们可以近似人类可能做出的变化。为了突出这种通过大型模型（ELM）进化的广泛意义，在主要实验中，ELM与MAPElites结合生成了数十万个Python程序的功能示例，这些程序输出Sodarace域中工作的移动机器人，而最初的LLM在预训练中从未见过。这些示例有助于引导训练新的条件语言模型，该模型可以为特定的地形输出正确的步行器。在以前没有训练数据的领域中，引导新模型的能力可以为给定的上下文输出适当的模型，这对open-endedness、深度学习和强化学习都有影响。本文对这些含义进行了深入探讨，以期启发ELM目前开辟的新的研究方向。

## 1 Introduction 介绍

For many in the evolutionary computation (EC) community, the rise of deep learning (DL) has raised questions on its implications for EC. Both approaches scale well with compute and both can yield useful discoveries and meaningful surprises. Yet are they ultimately competing paradigms, or rather are they complementary? In this paper we explore the latter possibility, of considerable synergy, by highlighting an untapped implication of large language models(LLMs; [1, 2]) for both genetic programming (GP; [3, 4]) and open-endedness[5–7].

对于进化计算（EC）社区的许多人来说，深度学习（DL）的兴起对其对EC的影响提出了质疑。这两种方法都可以很好地与计算机配合使用，并且都可以产生有用的发现和有意义的惊喜。然而，它们最终是相互竞争的范式，还是互补的？在本文中，我们通过强调大型语言模型（LLM；[1，2]）对于遗传编程（GP；[3，4]）和open-endedness[5-7]的一个未被利用的意义，探索了后一种可能，具有相当大的协同作用。

In particular, in this new Evolution through Large Models (ELM) approach, an LLM trained on code can suggest mutations that are intelligent, thereby facilitating a dramatically more effective mutation operator that sidesteps many of the challenges that previously existed for evolving programs [8]. Interestingly, the benefits of ELM are also reciprocal back to deep learning: the set of samples generated through the LLM can eventually constitute a new training set in a novel domain that can then fine-tune the LLM to perform well in the new domain, a novel data-generation procedure. Furthermore, this approach ultimately opens up new opportunities in the pursuit of open-endedness by increasing the generative capabilities of the LLM solely through its own generated data.

特别是在这种新的通过大型模型进化（ELM）的方法中，对代码进行训练的LLM可以提出智能化的突变建议，从而促进产生了一个明显更有效的突变算子，避开了许多以前存在的进化程序的挑战[8]。有趣的是，ELM的好处也是互惠的，回到深度学习：通过LLM产生的样本集最终可以构成一个新领域的新训练集，然后可以微调LLM，使其在新领域表现良好，这是一个新的数据产生程序。此外，这种方法最终在追求open-endedness方面开辟了新的机会，即仅仅通过LLM自己生成的数据来提高其生成能力。

LLMs have recently yielded impressive results in automated code generation [9, 10]. These models bootstrap from human knowledge by learning from very large datasets to achieve general coding competency. The fact that such bootstrapping is possible is clearly relevant to GP. After all, GP is in effect a generative approach to programming. While it might seem at first glance that LLMs therefore might out-compete or subsume GP, in fact GP does still offer an advantage in situations where the particular class of programs targeted by the search is far (or even completely lacking) from the training distribution of the LLM. In such cases, the LLM offers limited recourse (prompt engineering to learn an entirely new domain would be prohibitive), while GP can in principle evolve in any space (though in practice some spaces may be intractable due to the amount of mutation necessary to get consistent signal on fitness)

LLM最近在自动代码生成方面取得了令人印象深刻的结果[9, 10]。这些模型通过从非常大的数据集中学习，从人类的知识中引导出来，以达到一般的编码能力。这种引导是可能的，这一点显然与GP有关。毕竟，GP实际上是一种生成性的编程方法。虽然乍看之下，LLM可能会超越或取代GP，但事实上，在搜索所针对的特定程序类别与LLM的训练分布相去甚远（甚至完全没有）的情况下，GP仍然具有优势。在这种情况下，LLM提供了有限的资源（促使工程学习一个全新的领域将是令人望而却步的），而GP原则上可以在任何空间进化（尽管在实践中，由于获得一致的适应度信号所需的突变量，一些空间可能难以处理）。

Interestingly (and perhaps surprisingly), the best of both worlds is easily attainable by combining them: simply by prompting the LLM to generate changes the LLM can serve as a highly sophisticated mutation operator embedded within an overarching evolutionary algorithm. This way, the LLM in concert with evolution can steer each other towards the right region of the solution space even though neither evolution with a conventional mutation operator nor the LLM on its own could generate anything close.  In effect, program evolution using LLM-based perturbation begins to bridge the divide between evolutionary algorithms and those that operate on the level of human ideas. That is, LLMs can be trained to approximate how humans intentionally change programs, while staying on the manifold of functionality. Furthermore, such LLMs can be further fine-tuned on successful perturbations for the purposes of self-improvement, culminating in a novel technique for iteratively enhancing the performance of ELM.

有趣的是（也许令人惊讶的是），通过将二者结合起来，很容易达到两者的最佳效果：只需促使LLM生成更改，LLM就可以作为嵌入总体进化算法中的高度复杂的变异算子。这样，与进化相协调的线性规划可以将彼此引导到解空间的正确区域，即使使用传统变异算子的进化和线性规划本身都无法生成任何相近的结果。实际上，使用基于LLM的扰动的程序进化开始弥合进化算法和在人类思想水平上运行的算法之间的分歧。也就是说，LLM可以训练成近似人类如何有意更改程序，同时保持功能的多样性。此外，这种LLM可以在成功的扰动上进一步进行微调，最终形成一种新的技术，用于反复提高ELM的性能。

To highlight the potential of this approach, in this paper an entire dataset in a novel domain is generated from only a single mediocre starting example designed by hand by humans. In particular, the domain is called Sodarace[11, 12], where two-dimensional ambulating robots of arbitrary morphology are constructed for diverse terrains. Sodarace is cheap to simulate, allowing fast iteration, and also makes it easy to appreciate the sophistication of designs in-tuitively by simply watching the robot walk. In this way, it facilitates quick assessment of whether a design is successful both quantitatively and qualita-tively.

为了突出这种方法的潜力，在本文中，一个新领域中的整个数据集仅由人工设计的一个平庸的起始示例生成。特别是，该领域被称为Sodarace[11，12]，其中针对不同地形构建了任意形态的二维移动机器人。Sodarace模拟成本低，允许快速迭代，并且通过简单地观察机器人的行走，可以轻松地直观地欣赏设计的复杂性。通过这种方式，它有助于快速评估设计在数量和质量上是否成功。

To make the contribution of ELM explicit in the experiments in this paper, the Sodaracers are encoded as raw Python programs that output an enumeration of the ambulating robots’ components. That way, it is possible to demonstrate that ELM is a form of GP that can operate on a modern programming language directly, with no special provisions needed beyond the generic (i.e. not previously trained in Sodarace) existing code-generating LLM.

为了在本文的实验中明确ELM的贡献，Sodaracers被编码为原始Python程序，输出伏地机器人组件的枚举。这样，就有可能证明ELM是一种可以直接在现代编程语言上操作的GP方法，除了通用的（即以前没有在Sodarace中训练过）现有的代码生成LLM之外，不需要特殊的规定。

A final important insight unlocked by this approach is that the ability to generate diverse solutions in a domain or part of the search space where there was little to no training data is foundational to bootstrapping an open-ended process [6, 13, 14]. After all, open-endedness is fundamentally about searching outside the distribution of previous experience, which is exactly what ELM helps the LLM to do. Because this novel capability has potentially far-reaching implications, we have chosen in this work to focus on the implications of the generated data that can be produced by ELM. Of course, ELM is applicable in many other contexts that will undoubtedly be explored in the future.

这种方法的最后一个重要启示是，在几乎没有训练数据的或搜索空间的领域或部分产生不同解决方案的能力是引导开放式进程的基础[6, 13, 14]。毕竟，open-endedness从根本上说是在以前的经验分布之外进行搜索，这正是ELM帮助LLM所做的。因为这种新颖的能力有潜在的深远影响，所以我们在这项工作中选择将重点放在ELM所能产生的数据的影响上。当然，ELM也适用于许多其他情况，这些情况无疑将在未来被探索出来。

More specically, experiments that follow show that generated data is sufficiently rich that it can serve as training data for fine-tuning LLMs to generate code for viable new Sodaracers consistently, and furthermore that reinforcement learning (RL) can even fine-tune an augmented LLM to output Sodaracers conditionally, depending on the terrain. In the future, such conditional invention has the potential to unlock entirely new kinds of open-ended processes, just as humans have open-endedly built civilization over centuries by conditionally inventing its constituents.

更具体地说，接下来的实验表明，生成的数据足够丰富，可以作为微调LLM的训练数据，为可行的新Sodaracers持续生成代码，此外，强化学习（RL）甚至可以微调一个增强的LLM，根据地形有条件地输出Sodaracers。在未来，这种有条件的发明有可能开启全新的开放式过程，就像人类通过有条件地发明其组成部分，在几个世纪里以开放式方式构建文明一样。

In short, the main contributions of this paper are (1) the ELM method for efficiently evolving programs through LLMs, (2) a technique for improving ELM’s ability to search over time by ne-tuning its LLM-based mutation oper-ator, (3) a demonstration of ELM in a domain not included in the training data of the LLM, and (4) validation that data generated through ELM can bootstrap enhanced LLMs that offer a novel path towards open-endedness.

简而言之，本文的主要贡献是（1）通过LLM高效进化程序的ELM方法，（2）通过调整基于LLM的突变算子来提高ELM随时间搜索能力的技术，（3）在未包含在LLM训练数据中的域中演示ELM，（4）验证了通过ELM生成的数据可以引导增强的LLM，为open-endedness提供了一条新途径。

## 2 Background 背景

This section reviews previous work in genetic programming, large language mod-els, and open-endedness.

本节回顾了以前在遗传编程、大型语言模型和open-endedness方面的工作。

### 2.1 Genetic Programming 遗传编程

The field of genetic programming (GP) applies evolutionary algorithms to evolve computer programs to solve problems [3, 4, 15]. The promise of GP is that computer code is a computationally universal representation that underlies much modern technology, including artificial intelligence itself. Therefore it is conceivable for GP to automatically evolve programs that achieve human-level (or beyond) performance across diverse application domains [16]. However, there are obstacles in practice to its successful and widespread application to challenging problems.
 
遗传编程（GP）领域应用进化算法来进化计算机程序以解决问题[3, 4, 15]。GP的前景是，计算机代码是一种计算上的通用表示，它是许多现代技术的基础，包括人工智能本身。因此，可以想象，GP可以自动演化出在不同应用领域达到人类水平（或更高）的程序[16]。然而，在实践中，要成功和广泛地应用于具有挑战性的问题还存在一些障碍。 

One obstacle is that scaling GP to evolve increasingly complicated programs can be challenging [8], and that effectively applying GP to a new domain can require significant domain expertise. A researcher often must explicitly specify what functions, variables, and control structures are available to evolution [3, 17], which limits what ultimately can be evolved. In contrast, a human programmer can open-endedly decide what libraries to import and how to write many interdependent subroutines or classes. Research aims to lift these constraints, often through enabling modular reuse of code: e.g. through automatically de ned functions [3], data-mining populations to nd common sub-components [18], or attempts to use solutions to previous problems when solving new ones [19]. However, no method yet enables GP to scalably operate on human-designed programming languages with a minimum of domain-speci c tweaking.

其中一个障碍是，扩展GP以演化越来越复杂的程序可能是一个挑战[8]，并且有效地将GP应用到一个新的领域可能需要大量的领域专业知识。研究人员通常必须明确指定哪些函数、变量和控制结构可用于进化[3, 17]，这限制了最终可以进化的内容。相比之下，人类程序员可以无限制地决定导入哪些库以及如何编写许多相互依赖的子程序或类。研究的目的是解除这些限制，通常是通过实现代码的模块化重用：例如，通过自动定义的函数[3]、数据挖掘人口来发现共同的子组件[18]，或者试图在解决新问题时使用以前的解决方案[19]。然而，目前还没有一种方法能够使GP在人类设计的编程语言上进行可扩展的操作，而只需进行最少的领域特定调整。

A second obstacle is that nearly all GP methods explore through random perturbations of code, unlike humans, who through active practice improve their proficiency in making deliberate, complex, and coupled modifications to programs [20, 21]. Unlike perturbing e.g. neural network weights, wherein continuous parameters subject to small enough perturbations can predictably generate small changes in functionality [22, 23], perturbing code requires discrete changes that often dramatically shift functionality [24], thereby complicating search. While there exist approaches towards more directed generation offspring (e.g. building probabilistic models of high-performing programs [25], evolving repro-duction operators [26], or applying less-impactful mutation operators [24]), the problem remains at core unsolved.

第二个障碍是，几乎所有的GP方法都是通过对代码的随机扰动来进行探索的，这与人类不同，人类通过积极的实践来提高对程序进行有意的、复杂的和耦合的修改的熟练程度[20, 21]。与神经网络权重等的扰动不同，连续的参数受到足够小的扰动，可以预测地产生功能上的小变化[22, 23]，扰动代码需要离散的变化，往往会极大地改变功能[24]，从而使搜索复杂化。虽然有一些方法可以实现更多的定向生成后代（例如建立高性能程序的概率模型[25]，进化繁殖算子[26]，或应用影响较小的突变算子[24]），但这个问题的核心仍然没有解决。

In contrast to GP, humans learn to reason about code in its full complexity through experimentation and learning. This iterative effort leaves a permanent signature in repositories of code, such as GitHub. The next section describes progress in training large language models upon such repositories as a potential way to bypass the above obstacles.

与GP相比，人类通过实验和学习，学会了对代码的全部复杂性进行推理。这种反复的努力在代码库中留下了永久的签名，例如GitHub。下一节将介绍在这种资源库上训练大型语言模型的进展，作为绕过上述障碍的一种潜在方式。

### 2.2 Large Language Models 大型语言模型

Large language models (LLMs; [1, 2, 27]), trained on internet-scale data, have progressed at an impressive pace in recent years. The main idea (in auto-regressive models such as GPT-3 [2]) is to train increasingly-large neural net-works (built on the popular transformer architecture [28], sometimes with bil-lions of parameters) on the seemingly simple task of next-token prediction (i.e. given a sequence of tokens seen so far, predict the proceeding token). Scaling such LLMs (and formulating problems of interest as natural language processing tasks) has resulted in groundbreaking performance across a wide range of tasks[2, 29], including program synthesis [9, 10, 30].

近年来，在互联网规模的数据上训练的大型语言模型（LLMs; [1, 2, 27]）已经取得了令人瞩目的进展。主要的想法（在自动回归模型中，如GPT-3[2]）是训练越来越大的神经网络工程（建立在流行的变换器架构[28]上，有时有数十亿的参数）来完成看似简单的下一个标记预测的任务（即给定一个迄今为止看到的标记序列，预测下一个标记）。扩展这样的LLMs（并将感兴趣的问题制定为自然语言处理任务）已经在广泛的任务中产生了突破性的性能[2, 29]，包括程序合成[9, 10, 30]。

In particular, by training LLMs on large-scale coding data, e.g. from GitHub, it is possible to produce models with impressive function-synthesis capabilities [9, 10], highlighting the possibility to bootstrap the ability to fluently code from large-scale data. A further development are diff models that are trained on diffs from GitHub [31]. A diff is an incremental change to a file that is committed to a version control system such as GitHub, accompanied by a commit message describing the intent of the change. In this way, diff models are trained how, given a piece of code and any potential commit message, to suggest an informed change. Through the lens of evolutionary algorithms, such diff models can be viewed as intelligent perturbation operators, providing a way to walk over the manifold of code (in a controllable way) through mimicking human programmers. An interesting further possibility is that such models are amenable to further training through gradient descent, implying a potentially-powerful mechanism for self-adaptation (e.g. through reinforcing successful diffs during evolution). Both diff models and their capacity for self-adaptation are explored in this work as a way to improve GP. However, it is also important to note that general language models not trained directly on diffs can also act in effect like diff models when given the right kinds of prompts (see Section 3.1).

特别是，通过对大规模编码数据（例如来自GitHub）的LLM进行训练，可以生成具有令人印象深刻的函数合成能力的模型[9，10]，突出了从大规模数据中流畅编码的可能性。进一步的发展是基于GitHub[31]的差异训练的差异模型。差异是对提交到版本控制系统（如GitHub）的文件的增量更改，并伴随着描述更改意图的提交消息。通过这种方式，可以训练差异模型如何在给定一段代码和任何潜在的提交消息的情况下，提出明智的更改建议。从进化算法的角度来看，这种差异模型可以被视为智能扰动算子，通过模仿人类程序员，提供了一种在代码流形上行走（以可控方式）的方法。另一个有趣的可能性是，这种模型可以通过梯度下降进行进一步训练，这意味着一种潜在的强大的自适应机制（例如，通过在进化过程中加强成功的差异）。本文探讨了差分模型及其自适应能力，以此作为改进遗传算法的一种方法。然而，同样重要的是要注意到，在给予正确的提示时，没有直接针对差异进行训练的一般语言模型也可以像差异模型一样发挥作用（见3.1节）。

### 2.3 Open-endedness 开放性

With origins in the open-ended evolution community [6, 13, 32, 33] within artificial life, the field of open-endedness seeks to create algorithmic systems that produce never-ending innovation [5]. Given the primacy of search with ML, research within open-endedness naturally has focused on re ning algorithms for open-ended search, such as those driven by novelty [34, 35] or curiosity [36, 37]. While such focus has indeed lead to algorithmic progress, there is a growing awareness of the criticality of the environment in which open-ended algorithms are applied [38-41].

由于起源于人工生命中的开放式进化社区[6, 13, 32, 33]，open-endedness领域试图创建产生无止境创新的算法系统[5]。鉴于搜索在ML中的首要地位，开放性领域的研究自然而然地集中在重新确定开放性搜索的算法上，比如那些由新奇性[34, 35]或好奇心[36, 37]驱动的算法。虽然这种关注确实带来了算法上的进步，但人们越来越意识到应用open-endedness算法的环境的重要性[38-41]。

That is, the environment limits what can arise within the system and for how long its products can remain interesting. As a result, some have argued for more complex environments for open-endedness, such as video games [38, 39], and others have argued that features of the environment should co-evolve with agents [40, 42]. Yet a theory for what specific forms of additional such complexity is needed for enduring open-endedness has been lacking. This paper contributes a possible theory, arguing that agents outputting inventions into the environment in response to previous inventions may be a principled route to such continuing open-endedness.

<!-- TODO  -->
也就是说，环境限制了系统内可以产生的东西，以及其产品可以保持多长时间的趣味。因此，一些人主张为open-endedness提供更复杂的环境，如视频游戏[38, 39]，另一些人主张环境的特征应该与代理共同进化[40, 42]。然而，关于持久的open-endedness需要什么具体形式的额外的这种复杂性的理论一直是缺乏的。本文提供了一个可能的理论，认为代理人根据以前的发明向环境输出发明可能是实现这种持续open-endedness的原则性途径。

One challenge in evolving aspects of the environment (such as inventions), is how they are encoded. Most research applies encodings that are specifically dit to describe some fixed part of a larger environment, e.g. a fixed way of describing edges within a maze [43], or the shape of a 2-D landscape [40]. While sometimes the encodings of these parts are universal (e.g. the CPPN encoding of landscapes in [40] can describe any landscape, and the RNN encoding of Dennis et al. [42] can describe any maze), it is unclear how to expand the representation to include more of the environment without relying upon ad-hoc principles. This paper argues that computer programs are a general and powerful encoding for continually expanding the richness of an existing environment.

在进化环境的各个方面（如发明）的一个挑战是如何对它们进行编码。大多数研究应用的编码是专门用来描述更大环境的某些固定部分的，例如，描述迷宫内的边缘的固定方式[43]，或者二维景观的形状[40]。虽然有时这些部分的编码是通用的（例如，[40]中景观的CPPN编码可以描述任何景观，Dennis等人的RNN编码[42]可以描述任何迷宫），但目前还不清楚如何在不依赖特别原则的情况下扩展表示方法以包括更多的环境。本文认为，计算机程序是一种通用的、强大的编码，可以不断地扩展现有环境的丰富性。

## 3 Approach: Evolution through Large Models 方法：通过大型模型的演变

Three distinct components facilitate ELM. First is the novel mutation operator driven by an LLM. Second is an evolutionary outer loop that calls this mutation operator. Finally, the third component is a method for updating the LLM to improve based on its preceding performance. Each of these is detailed in this section.

三个不同的组件促进了ELM。首先是由LLM驱动的新型突变算子。其次是调用该突变算子的进化外循环。最后，第三个组成部分是一种更新LLM的方法，以根据其先前的性能进行改进。本节将详细介绍其中的每一项。

### 3.1 Mutation through Diff 通过差异的突变

The main idea behind ELM centers on rethinking the mutation operator for code by exploiting the capabilities of LLMs. In conventional GP, the language of the code and the types of changes allowed through mutation are both chosen intentionally to yield a reasonable chance that perturbations can lead to useful functional changes [3]. In contrast, LLMs unlock an entirely di erent basis for mutation: it would be more ideal if the mutation operator understood the code and how it can be changed in interesting ways, more like a human than a stochastic event.

ELM的主要思想是通过利用LLM的能力来重新思考代码的变异操作。在传统的GP中，代码的语言和通过突变允许的变化类型都是有意选择的，以产生合理的机会，使扰动能够导致有用的功能变化[3]。相比之下，LLMs为突变提供了一个完全不同的基础：如果突变操作者了解代码以及如何以有趣的方式改变它，更像一个人而不是一个随机事件，那就更理想了。

LLMs can indeed be trained to output code in an autoregressive manner by exposing them to extensive programming examples [9, 10]. A diff model [31] can similarly be autoregressively trained on a collection of code diffs (e.g. from GitHub). Each diff targets a single file, where the file and diff are short enough to fit into the context of the LLM. The model is trained to predict the diff (formatted, for example, in unified diff format [44]) from the concatenation of the file and the commit message, where the loss includes only the tokens that make up the diff , thereby encouraging the model to predict the diff but not to memorize the file and commit message. In other words, the model learns to predict plausible changes to code from examples of changes made to code by human programmers. It is important to note that the idea of diff models (or their initial training) [31] is not a contribution of this paper, but diff models are rather a tool applied here in a new context (to produce mutations).

LLMs确实可以通过让它们接触大量的编程实例来训练它们以自回归的方式输出代码[9, 10]。差异模型[31]同样可以在一系列的代码差异（例如来自GitHub）上进行自回归训练。每个差异都针对一个文件，文件和差异都很短，足以适应LLM的上下文。模型被训练成从文件和提交信息的串联中预测差异（格式化，例如，统一的差异格式[44]），其中损失只包括构成差异的标记，从而鼓励模型预测差异，但不记忆文件和提交信息。换句话说，该模型学会了从人类程序员对代码进行修改的例子中预测合理的修改。需要注意的是，diff模型的思想（或其初始训练）[31]并不是本文的贡献，而是diff模型在这里是一个应用于新环境的工具（产生突变）。

To achieve meaningful mutations, ELM can choose among a set of commit messages, which convey to the LLM the details of the operation it should per-form in lieu of mutation. These messages o er signi cant power and nuance for calibrating mutation operators that is likely highly novel to anyone familiar with implementing mutation in GP or evolutionary algorithms in general. In the experiment in this paper, the three commit messages and their respective probabilities of being chosen are:

为了实现有意义的突变，ELM可以在一组提交信息中进行选择，这些信息向LLM传达它应该执行的操作的细节，以代替突变。这些信息对于校准突变操作者来说具有重大意义，对于任何熟悉在GP或一般进化算法中实现突变的人来说，这可能是非常新颖的。在本文的实验中，三个提交信息及其各自被选择的概率是：

- Changed make walker function. (40% chance) 更改了make walker函数。（40%的可能性）

- Changed parameters in make walker function. (30% chance) 更改了make walker函数中的参数。（30%的可能性）

- Small change to make walker function. (30% chance) 对make walker函数进行小改动。（30%的可能性）


Of course, any commit message is conceivable. The LLM’s ability to interpret general natural language means that the scope for research exploration (and domain-speci city) here is vast.

当然，任何提交信息都是可以想象的。LLM解释一般自然语言的能力意味着这里的研究探索（和特定领域的研究）的范围很广。

As a simple experiment to highlight diff models’ ability to intelligently modify code, an implementation of a function with an adjustable amount of bugs is perturbed with either a simple GP mutation operator or with a 300M parameter diff model. The hypothesis is that an intelligent perturbation operator will be better able to make multiple correlated changes to code (in this case to correct the bugs). The 4-Parity task (which is inspired by a standard GP benchmark [3]) serves as a representative test-bed. Note that a correct implementation of 4-Parity returns the sum of the four input bits, modulo two. Up to five bugs are introduced to 4-Parity, first by incrementally misnaming each of the variables in the sum calculation; and for the fifth bug, the modulo is changed from two to three. Then, perturbation operators are tested for their ability to (in one perturbation step) change the buggy version of the code to one that successfully passes unit tests. Results in gure 1 highlight how with increasing bugs GP mutation becomes exponentially more unlikely to produce a successful solution (note that no mutation from GP solves all five, given 100,000 trials). In contrast, the diff operator is able to fix all five bugs, and its performance is impacted more by the number of different types of bugs (i.e. the fifth bug affects the modulo calculation rather than renaming variables) than by the raw number of bugs itself. Further details (including a supporting experiment with another task with similar results) are given in Appendix A.

作为强调差异模型智能修改代码能力的简单实验，具有可调bug数量的函数的实现会受到简单GP变异算子或300M参数差异模型的干扰。假设智能扰动算子能够更好地对代码进行多个相关更改（在本例中是为了纠正错误）。4-Parity任务（受标准GP基准的启发[3]）作为一个代表性的测试平台。请注意，4-Parity的正确实现会返回四个输入位的总和，并以2为模数。在4-Parity中引入了多达五个错误，首先是在计算总和的过程中逐步错误地命名每个变量；对于第五个错误，模数从2改为3。然后，对扰动运算符的能力进行测试，看其是否能够（在一个扰动步骤中）将代码的错误版本改为能够成功通过单元测试的版本。图1中的结果强调了随着bug的增加，GP突变产生成功解决方案的可能性呈指数增长（注意，在100,000次试验中，没有GP突变能解决所有五个问题）。相比之下，差异算子能够修复所有五个bug，其性能更多的是受到不同类型bug数量的影响（即第五个bug影响到模计算，而不是重命名变量），而不是受到原始bug数量本身的影响。进一步的细节（包括用另一个任务进行的具有类似结果的辅助实验）在附录A中给出。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%201.png"/></div>

> Figure 1: Comparing diff mutation to GP mutation in fixing 4-Parity bugs. The figure shows how the ability of a single mutation to produce correct solutions changes as bugs are incrementally added to a working 4-Parity implementation. Note that success percentage is shown in log scale, i.e. success for GP mutation decreases exponentially in the number of mutations (and produces no solutions when there are five bugs). In contrast, diff mutation degrades only with the fifth bug. The conclusion is that LLM-based mutation can indeed make multiple sensible coupled changes to code. \
图1：比较差异突变和GP突变在修复4-Parity错误方面的作用。该图显示了当错误被逐步添加到工作的4-Parity实现中时，单一突变产生正确解决方案的能力如何变化。请注意，成功率是以对数比例显示的，也就是说，GP突变的成功率随着突变次数的增加而呈指数级下降（当有五个错误时，不产生任何解决方案）。相比之下，diff突变只有在出现第五个错误时才会退化。结论是，基于LLM的突变确实可以对代码进行多次合理的耦合修改。

Because the tools involved in an ELM implementation are unconventional, we finally wanted to highlight here several alternatives for implementing such systems in practice today. One option is to use models available on the OpenAI API that can edit through following instructions [45, 46]. A second option is to create an intelligent mutation operator through few-shot prompting instead of through explicit training (as in the di model). That is, one could design prompts for a model trained on code (like Codex [9] or GPT-6-J [47]). To show the potential to replicate (or improve upon) the results in this paper, we conducted a simple experiment comparing (on the 4-Parity problem) prompt engineering and edit mode to the di model. Figure 2 shows how models from the API outperform the di model used in the paper. Further experimental details can be found in Appendix A.

因为ELM的实施所涉及的工具是非常规的，所以我们最后想在这里强调一下今天在实践中实施这种系统的几个替代方案。一种选择是使用OpenAI API上的模型，它可以通过跟随指令进行编辑[45, 46]。第二个选择是通过少量的提示，而不是通过明确的训练（如差异模型）来创建一个智能突变操作者。也就是说，人们可以为一个在代码上训练的模型（如Codex[9]或GPT-6-J[47]）设计提示语。为了显示复制（或改进）本文结果的潜力，我们进行了一个简单的实验，将（在4-Parity问题上）提示工程和编辑模式与差异模型进行比较。图2显示了来自API的模型如何胜过本文中使用的差异模型。进一步的实验细节可以在附录A中找到。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%202.png"/></div>


> Figure 2: Comparing alternate LLM-based mutations in fixing 4-Parity bugs. The performance of different mutation operators in fixing bugs is shown as bugs are incrementally added to a working 4-Parity implementation. Both edit mode and prompt-engineering approaches outperform the diff model applied in this paper’s experiments. The conclusion is that both prompt-engineering on LLMs trained on code and using edit mode models from the OpenAI API are viable options to build upon the work in this paper. \
图2：比较基于LLM的变异在修复4-Parity错误方面的作用。不同的突变操作者在修复错误方面的表现是随着错误被逐步添加到一个工作的4-Parity实现中而显示的。编辑模式和提示工程方法都优于本文实验中应用的差异模型。结论是，在代码上训练的LLM的提示工程和使用OpenAI API的编辑模式模型都是在本文工作基础上的可行选择。

### 3.2 The Evolutionary Algorithm and Implications for Open-Endedness  进化算法和对Open-Endedness的影响

Because the mutation operator is effectively a modular component for many evolutionary algorithms [48, 49], ELM can be implemented within a diversity of contexts. Of course, the approach is most applicable to a case where the genetic encoding is through a known programming language, because that is how the benefits of the LLM will be realized. Genetic encodings in natural language or any other language at which LLMs excel are also conceivable, but of course the utility of such encodings would depend on how they are applied and their mapping to a potentially useful phenotype. The experiments in this paper focus on Python 3 genotypes, which are also by their nature variable in length. The ability to use modern programming languages as genotypes without the need for any special accommodation is a key benefit of ELM.

因为变异算子实际上是许多进化算法的一个模块化组件[48, 49]，ELM可以在多种情况下实施。当然，该方法最适用于遗传编码是通过一种已知的编程语言的情况，因为这就是LLM的好处将被实现的方式。用自然语言或任何其他LLM擅长的语言进行遗传编码也是可以想象的，当然，这种编码的效用将取决于如何应用它们以及它们与潜在的有用表型的映射。本文的实验集中在Python 3基因型上，这些基因型的长度也是可变的。使用现代编程语言作为基因型的能力，而不需要任何特殊的适应性，这是ELM的一个关键好处。

While there are many options for the evolutionary algorithm in the outer loop, we have chosen in this paper to implement ELM within a quality diversity (QD) algorithm [50, 51]. An important motivation for this choice is that the emergence of the ability to search intelligently for arbitrarily complex programs is tantalizingly close to overcoming some of the key obstacles to open-endedness [14], and ELM is an opportunity to highlight this opportunity.

虽然外环中的进化算法有许多选项，但我们在本文中选择在质量多样性（QD）算法[50，51]中实现ELM。这一选择的一个重要动机是，智能搜索任意复杂程序的能力的出现非常接近于克服open-endedness的一些关键障碍[14]，ELM是一个强调这一机会的机会。

Recall that we do not yet know how to make an algorithm that exhibits genuinely open-ended divergence. While there has been progress towards open-endedness in recent years, the state of the art remains weak open-endedness, wherein novel and interesting discovery continues only for a brief time, eventually ending in a plateau as the possibilities are exhausted [5, 40, 43, 52-54]. In contrast, in strong open-endedness, the process would never plateau-if we left and returned a year later, or even a million years later, its products would continue to become more interesting over time. No algorithm comes close to such an achievement, though it is evidently possible in nature.

回顾一下，我们还不知道如何制作一个表现出真正的open-endedness发散的算法。虽然近年来在open-endedness方面取得了进展，但目前的技术状况仍然是弱open-endedness，其中新奇有趣的发现只持续了很短的时间，最终随着可能性的耗尽而结束在plateau上[5, 40, 43, 52-54]。相比之下，在强open-endedness中，这个过程永远不会出现plateau现象--如果我们离开，一年后甚至一百万年后再回来，它的产品会随着时间的推移继续变得更加有趣。没有任何算法能接近这样的成就，尽管它在自然界显然是可能的。

The question then is what stands between today’s algorithms and tractable strong open-endedness. This gap remains despite that recent work in open-endedness appears to make progress. For example, the Enhanced POET algo-rithm continues to generate diverse and increasingly complex terrains for bipedal robots to solve [40]. In their hide-and-seek experiment, Baker et al. [54] show agents discovering increasingly complex strategies like assembling blocks into a hideout. Yet despite such algorithms clearly demonstrating the capability to continue to invent new solutions, all such demonstrations share a singular downfall: they slow down and eventually end. Formalizing ELM within a QD framework in e ect o ers a novel opportunity to address this challenge.

那么问题来了，今天的算法和可操作的强open-endedness之间有什么差距。尽管最近在open-endedness方面的工作似乎取得了进展，但这个差距仍然存在。例如，增强型POET算法继续为双足机器人生成各种不同的、越来越复杂的地形来解决[40]。在他们的捉迷藏实验中，Baker等人[54]展示了代理人发现越来越复杂的策略，如将积木组装成一个藏身处。然而，尽管这些算法清楚地展示了继续发明新解决方案的能力，但所有这些演示都有一个共同的缺点：它们的速度变慢并最终结束。在QD框架内将ELM正式化，为解决这一挑战提供了新的机会。

This opportunity connects to the difficulty of formulating an artificial environment that imposes no limit on what even the most capable open-ended algorithm can achieve, as noted in the Background. The challenge of devising artificial environments with unbounded potential raises the intriguing question of what property our universe and planet possess that is lacking in current artificial environments. This question is critically important for open-endedness because in the absence of that property, open-ended algorithms cannot demonstrate their full potential. If the problem indeed stems from the fact that artificial environments to date offer only finite possible experiences until their potential is exhausted, then to overcome this bottleneck the environment itself needs to possess the potential to change forever.

如背景中所述，该机会与制定人工环境的难度有关，该人工环境对即使是最有能力的open-ended算法也可以实现的目标没有限制。 设计具有无限潜力的人工环境的挑战提出了一个有趣的问题，即我们的宇宙和地球拥有当前人工环境所缺乏的属性。 这个问题对于open-endedness至关重要，因为在没有该属性的情况下，open-endedness算法无法展示其全部潜力。 如果问题确实源于人工环境迄今为止只能提供有限的可能体验，直到它们的潜力被耗尽，那么为了克服这个瓶颈，环境本身就需要拥有永远改变的潜力。

Since the emergence of intelligence in nature, much environmental change has been driven by the intelligent agents themselves. Eventually, humans acquired the ability to leave detached artifacts in the environment that radically alter its potential for themselves and other agents, like a house, a vehicle, or even a program. Unlike new organisms that are evolved over generations, such detached conditional things (DCTs) are generated intentionally as a condition of the observations of the agent. Once DCTs enter the world, open-endedness accelerates because the environment is rapidly updating even within the course of a single lifetime.

自从自然界中出现智能以来，许多环境变化都是由智能代理本身驱动的。 最终，人类获得了将分离的人工制品留在环境中的能力，这从根本上改变了它对自己和其他代理的潜力，如房屋、车辆甚至程序。 与经过几代人进化的新生物不同，这种分离的条件事物（DCT）是有意生成的，作为代理观察的条件。 一旦 DCT 进入世界，open-endedness就会加速，因为即使在一个生命周期内环境也在迅速更新。

Each DCT creates an opportunity for further DCTs. For example, the invention of the door creates the opportunity for keys to be invented, which then sets the stage for lock picks, and so on. And because they are detached, DCTs can leave a permanent legacy in the environment well beyond the lifetime of their inventor. In this way, invention in the era of DCTs is open-ended, and accordingly has continued for thousands of years, from fire and wheels to space stations and computers.

每个DCT都为进一步的DCT创造了机会。例如，门的发明为钥匙的发明创造了机会，而钥匙的发明又为开锁创造了条件，如此类推。由于它们是分离的，DCTs可以在环境中留下永久的遗产，远远超过其发明者的寿命。这样一来，DCT时代的发明是无止境的，相应地，从火和车轮到空间站和计算机，已经持续了数千年。

This theory of DCTs supplies an abstract answer to the problem of a limited environment: Agents must be able to imprint the environment with DCTs in response to those already present within it. However, realizing DCTs in practice requires addressing a separate question: how can agents be enabled to e ciently invent DCTs of limitless complexity in a new domain?

这个DCTs理论为有限环境的问题提供了一个抽象的答案。代理人必须能够用DCTs对环境中已经存在的DCTs做出反应。然而，在实践中实现DCTs需要解决另一个问题：如何使代理人能够在一个新的领域中轻松地发明无限复杂的DCTs？

Interestingly, computer programs are universal representations, meaning that the procedure of assembling new artifacts can naturally be described algorithmically. For example, programmers have leveraged code to help create enormously complex artifacts (like the layouts of computer chips or instructions for 3-D printers to produce complex physical objects). Of course, programs themselves can function as DCTs. In this way, a procedure that can search through modern program space and ultimately generate such programs conditionally is a candidate for creating open-ended environments of unlimited capacity. The experiment in this paper will demonstrate in more detail how ELM makes such a construct conceivable; the significance of QD is that its ability to generate a diverse space of artifacts can serve as the bootstrap to obtaining agents capable of generating DCTs. In short, the QD algorithm is generating training data that can transform the LLM into a kind of DCT generator.

有趣的是，计算机程序是通用的表征，这意味着组装新人工制品的程序自然可以用算法来描述。例如，程序员利用代码来帮助创造极其复杂的人工制品（如计算机芯片的布局或3-D打印机生产复杂物理物体的指令）。当然，程序本身也可以作为DCT发挥作用。这样一来，一个能够在现代程序空间中搜索并最终有条件地生成这种程序的程序是创造无限容量的开放式环境的候选者。本文的实验将更详细地证明ELM是如何使这样的构造成为可能的；QD的意义在于，它生成多样化人工制品空间的能力可以作为获得能够生成DCT的代理的引导。简而言之，QD算法产生的训练数据可以将LLM转化为一种DCT发生器。

While any QD algorithm can work with ELM, the specific algorithm in the experiment in this paper is MAP-Elites [51, 55] (Figure 3). The core of MAP-Elites is a uniformly-spaced grid of niches (called the map), that spans user-specified dimensions of solution diversity, called the behavior characterization. Upon initialization, a single pre-existing (hand-designed in this paper) solution is evaluated and placed into the map. In each iteration thereafter, an inhabited niche is randomly chosen and the solution within that niche is perturbed by the diff model and evaluated. The new candidate solution is assigned its niche from its behavior characterization, and if that niche is unfilled or the new solution out-performs the niche’s current inhabitant, it becomes the champion of that niche; otherwise, the candidate is discarded. In this way, over iterations of search, the map gradually fills with increasingly diverse and high-quality solutions.

虽然任何 QD 算法都可以与 ELM 一起使用，但本文实验中的特定算法是 MAP-Elite [51, 55]（图 3）。 MAP-Elite 的核心是一个均匀间隔的生态位网格（称为地图），它跨越了用户指定的解决方案多样性维度，称为行为表征。 初始化后，评估单个预先存在的（本文中手工设计的）解决方案并将其放入地图中。 在此后的每次迭代中，随机选择一个有人居住的生态位，并且该生态位内的解决方案受到差异模型的干扰并进行评估。 新的候选解决方案根据其行为特征分配其生态位，如果该生态位未填充或新解决方案的表现优于该生态位的当前居民，它将成为该生态位的冠军； 否则，候选人被丢弃。 这样，在搜索的迭代中，地图逐渐充满了越来越多样化和高质量的解决方案。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%203.png"></div>
> Figure 3: MAP-Elites with ELM. In each iteration, an existing Python solution is sampled from the map of solutions for each independent replica of a diff model. Each replica generates a batch of diffs that are applied to the sampled solution to generate modified candidate solutions. These candidates are evaluated and are then inserted into the map if they either establish a new niche or outperform the niche’s current champion. Over iterations, a single initial seed program gives rise to a diversity of high-performing Python programs. \
图 3：使用 ELM 的 MAP-Elite。 在每次迭代中，现有的 Python 解决方案都是从差异模型的每个独立副本的解决方案映射中抽取的。 每个副本都会生成一批差异，这些差异应用于采样解决方案以生成修改后的候选解决方案。 这些候选人会被评估，然后如果他们建立新的生态位或优于该生态位的当前冠军，则将其插入地图。 在迭代过程中，单个初始种子程序会产生多种高性能 Python 程序。

### 3.3 Fine-tuning the Diff Operator 微调差分算子

Interestingly, because the mutation (diff) operator is itself an LLM, it has the potential to be improved with respect to the domain. While self-adaptation [56-58] has a long tradition in evolutionary computation, including algorithms suchas CMA-ES [58] and natural evolution strategies [59], the kinds of improvements possible in ELM are unique by offering the possibility of the LLM learning how to think about change. That is, ideas for changes that are most promising in one domain might be different than in another, and the richness of the LLM offers the potential to capture such nuance through experience. In particular, the pre-trained diff model can be trained further (which is called fine-tuning) with accepted diffs (by MAP-Elites) from initial iterations or runs of ELM. That way, the diff operator updates to understand better the kinds of modifications that lead to either higher quality, more novelty, or both. This fine-tuning technique can cause ELM itself to improve over iterations. Of course, over a long run, the ideal kinds of changes might change; continually fine-tuning based on recent experience can potentially track such drifting opportunities. In this paper, the potential of fine-tuning is demonstrated through a single fine-tuning iteration, but the investigation of such continual refinement is an open research opportunity. Note that the prompt-engineering approach to LLM mutation described at the end of Section 3.1 can also bene t from fine-tuning in this way.

有趣的是，由于突变(diff)算子本身是一个LLM，它有可能在领域方面得到改进。虽然自我适应[56-58]在进化计算中有着悠久的传统，包括CMA-ES[58]和自然进化策略[59]等算法，但ELM中可能的改进种类是独特的，因为它提供了LLM学习如何思考变化的可能性。也就是说，在一个领域中最有希望的变化想法可能与另一个领域不同，而LLM的丰富性提供了通过经验捕捉这种细微差别的可能性。特别是，预训练的差异模型可以通过ELM的初始迭代或运行中接受的差异（由MAP-Elites）进一步训练（这被称为微调）。这样一来，差异操作者就能更好地理解导致更高的质量、更多的新颖性或两者兼而有之的各种修改。这种微调技术可以使ELM本身在迭代中得到改善。当然，在长期运行中，理想的修改种类可能会发生变化；根据最近的经验不断地进行微调，有可能跟踪这种漂移的机会。在本文中，微调的潜力是通过一个单一的微调迭代来展示的，但对这种持续的细化的调查是一个开放的研究机会。请注意，第3.1节末尾描述的LLM突变的提示工程方法也可以通过这种方式从微调中受益。

## 4 Experiment and Results 实验与结果

The primary motivation for the experiment that follows is to give a taste of the breadth of implications of ELM, to evolutionary computation, to deep learning, and to open-endedness. For this purpose, this experiment focuses on the prob-lem of the invention of complex artifacts (which could eventually serve as DCTs in a future more ambitious experiment). While the potential scope of applica-tions for ELM is broad, the opportunity to learn to invent complex artifacts in an arbitrary domain extends directly from the augmented ability to search through programs; seeing this inventive capability realized thereby highlights novel opportunities opening up.

以下实验的主要动机是体验 ELM 的在进化计算、深度学习和open-endedness广泛影响。 为此，本实验重点关注复杂人工制品的发明问题（最终可能在未来更雄心勃勃的实验中用作 DCT）。 虽然 ELM 的潜在应用范围很广，但学习在任意领域发明复杂工件的机会直接来自于增强的程序搜索能看，到这一发明能力的实现，突显了新的机遇。

The experiment will aim to bootstrap from a few hand-written (and barely functional) examples of an invention into an LLM-based inventor that can fluidly output appropriate inventions conditioned on its environment. This concept is demonstrated in the domain of Sodarace [11, 12], a physics-based invention domain that serves as a cheap-to-simulate microcosm of invention. The goal in Sodarace is to construct from collections of masses and oscillating springs two-dimensional robots that can locomote competently. A wide range of interesting Sodaracer robots are possible, as highlighted by previous ML research [12] and the origins of the domain: Sodarace began as a web application called Sodacon-structor, wherein the human design of Sodaracers was su ciently compelling for an online community to form around the endeavor [11].

该实验旨在从一些手写（几乎没有功能）的发明示例引导到基于 LLM 的发明者，该发明者可以根据其环境流畅地输出适当的发明。 这个概念在 Sodarace [11, 12] 的领域中得到了证明，这是一个基于物理的发明领域，可用作廉价模拟发明的缩影。 Sodarace 的目标是从质量块和振动弹簧的集合中构建出能够有效移动的二维机器人。 正如之前的 ML 研究 [12] 和该领域的起源所强调的那样，各种有趣的 Sodaracer 机器人都是可能的 社区围绕努力形成[11]。

An individual Sodaracer (Figure 4) is composed of a variable-sized collection of point masses (each fully-described by its initial 2-D position) and oscillating springs that connect masses together. The motion of the robot is driven by the oscillation of its springs, and each spring has parameters specifying the amplitude and phase of its oscillation (by convention all springs have the same period). To evaluate a particular Sodaracer, it is simulated in a specific terrain for a fixed amount of time and its ability to traverse that terrain is measured (i.e. how far the Sodaracer’s center of mass moves along the fix-axis); additionally, to measure the diversity of solutions for MAP-Elites, features capturing gross aspects of the robot’s morphology (i.e. its initial height, width, and total mass) are recorded. While a search algorithm could operate directly in the space of masses and springs, here instead LLMs output Python code that describes the morphology of the Sodaracer. For examples of such source code, see Appendix B and G. In this way, the programs evolved by ELM are in effect indirect encodings [60-63] for Sodaracers. That is, in principle any indirect encoding expressible through code could be evolved from scratch or modified by ELM.

一个单独的Sodaracer（图4）由一个大小可变的点质量集合（每个点质量由其初始二维位置完全描述）和将质量连接在一起的振荡弹簧组成。机器人的运动是由其弹簧的振荡驱动的，每个弹簧都有指定其振荡的振幅和相位的参数（按照惯例，所有弹簧都有相同的周期）。为了评估一个特定的Sodaracer，它在一个特定的地形中模拟了一段固定的时间，并测量了它穿越该地形的能力（即Sodaracer的质心沿固定轴移动了多远）；此外，为了测量MAP-Elites解决方案的多样性，记录了捕捉机器人形态的粗略方面的特征（即其初始高度、宽度和总质量）。虽然搜索算法可以直接在质量和弹簧的空间中操作，但在这里，LLMs输出的Python代码描述了Sodaracer的形态。这样一来，ELM所演化的程序实际上是Sodaracers的间接编码[60-63]。也就是说，原则上任何可通过代码表达的间接编码都可以从头开始演化或由ELM修改。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%204.png"></div>

> Figure 4: An Example Sodaracer. The objective in the Sodarace domain is to design a Sodaracer that locomotes e ectively across the ground terrain. Labeled in the image are examples of a mass and a spring that connects two masses together. A Sodaracer design consists of a variable number of masses and springs, where springs also have oscillatory parameters that determine the Sodaracer’s motion. \ 图 4：Sodaracer 示例。 Sodarace 领域的目标是设计一种能够有效地穿越地面地形的 Sodaracer。 图像中标记的是质量和将两个质量连接在一起的弹簧的示例。 Sodaracer 设计由数量可变的质量块和弹簧组成，其中弹簧还具有决定 Sodaracer 运动的振荡参数。

More ambitiously than only generating a repertoire of Sodaracer designs, the experiment will attempt to implement an entire invention pipeline that ultimately yields a novel kind of conditional LLM that can input a terrain and output an appropriate Sodaracer for that terrain. ELM thereby serves as the initial data generation phase of this pipeline, showing in this way how ELM can serve in general as a way of generating domain data for downstream deep learning where it did not previously exist. Furthermore, in the future the ability to train such conditional inventors could serve as a foundation for an open-ended world of DCT-generating agents.

比起仅仅生成一个Sodaracer设计剧目，该实验将尝试实现整个发明流水线，最终产生一种新型的条件性LLM，它可以输入一个地形并为该地形输出一个合适的Sodaracer。因此，ELM作为这个流水线的初始数据生成阶段，以这种方式展示了ELM如何在一般情况下作为下游深度学习的领域数据的生成方式，而这种数据以前并不存在。此外，在未来，训练这种条件性发明者的能力可以作为一个开放的DCT生成代理世界的基础。

In practice, the aim of the invention pipeline is to create an agent that can output complex artifacts conditionally, based on its observation of the environment. If invention is conceived as an action, then learning to invent conditionally can be viewed as a reinforcement learning (RL) problem [64]. That is, for any given observation, the agent can be rewarded depending on the success of the resultant invention. For example, in Sodarace, the agent might observe a speci c terrain such as a hill and then output a design for a Sodaracer arti-fact. The reward then depends upon the performance of the Sodaracer in the observed terrain.

在实践中，发明流水线的目的是创建一个能够根据其对环境的观察有条件地输出复杂人工制品的代理。如果发明被认为是一种行动，那么学习有条件地发明可以被看作是一个强化学习（RL）问题[64]。也就是说，对于任何给定的观察，代理人可以根据结果发明的成功与否而得到奖励。例如，在Sodarace中，代理可能观察到一个特定的地形，如山丘，然后输出一个Sodaracer人工智能的设计。然后，奖励取决于Sodaracer在所观察到的地形中的表现。

This approach sounds straightforward-it is simply RL with complex outputs but there is a problem. If the agent has no prior experience in the domain (e.g. in Sodarace), then outputting even a valid (let alone working) artifact is e ectively impossible. As a result, there is no gradient for RL and it cannot bootstrap into the new domain.

这种方法听起来很直接--它是简单的RL与复杂的输出，但有一个问题。如果代理在该领域没有任何经验（例如在Sodarace），那么即使输出一个有效的（更不用说工作）工件也是不可能的。因此，RL没有梯度，它不能引导到新的领域。

Therefore, to get RL started, some form of pretraining is necessary. In effect, the RL fine-tuning described above is actually the last step in a pipeline, where the preceding step is to teach the agent something preliminary about its domain. For that purpose, an LLM can be trained on a large set of examples from the target domain. For example, these examples could be Sodarace walker designs. After exposure to enough such designs, in principle the LLM knows something about the domain and can output sample artifacts from the training distribution. With such knowledge later passed on to RL, it should now be possible to bootstrap into conditional fine-tuning.

因此，要开始强化学习，某种形式的预训练是必要的。 实际上，上面描述的 RL 微调实际上是流水线中的最后一步，其中前面的步骤是教代理一些关于其领域的初步知识。 为此，可以使用来自目标域的大量示例来训练 LLM。 例如，这些示例可能是 Sodarace walker设计。 在接触到足够多的此类设计之后，原则上 LLM 对领域有所了解，并且可以从训练分布中输出样本工件。 随着这些知识后来传递给 RL，现在应该可以引导进入条件微调。

However, there is still a problem: where did all the examples come from for training the LLM? If the hope is for the conditional inventor eventually to invent in a novel domain like Sodarace where a generic LLM is unlikely to have any exposure, then the source for all the examples needed to train the LLM is itself elusive. As a consequence, the pipeline needs yet one more preceding step-which is where ELM comes in-to generate a set of example artifacts from scratch, which could then train the LLM that will eventually bootstrap RL.

但是，仍然存在一个问题：所有用于训练 LLM 的示例都来自哪里？ 如果希望有条件的发明者最终在像 Sodarace 这样的新领域中进行发明，其中通用 LLM 不太可能有新领域的任何有关的信息，那么训练 LLM 所需的所有示例的来源本身就是难以捉摸的。 因此，流水线还需要一个更前面的步骤——这就是 ELM 的用武之地——从头开始生成一组示例工件，然后可以训练最终引导 RL 的 LLM。

Generating a diverse and large set of initial training examples is a search problem. However, because no LLM yet has any exposure to the right kind of data, it is a search problem within the invention space rather than within the weight space of neural networks. Searching for diverse functional examples (to get a wide pre-training distribution) within the space of artifacts is the natural role of QD (i.e. MAP-Elites in this paper). Combined with the diff function, the result is ELM, which yields a novel approach to generating training examples, thereby bootstrapping the entire process.

生成一个多样化的大型初始训练实例集是一个搜索问题。然而，由于目前还没有LLM接触到合适的数据，所以这是一个在发明空间内而不是在神经网络的权重空间内的搜索问题。在人工制品的空间内搜索不同的功能实例（以获得广泛的预训练分布）是QD（即本文中的MAP-Elites）的自然作用。与差异函数相结合，其结果是ELM，它产生了一种新的生成训练例子的方法，从而引导了整个过程。

To recap, what emerges is a three-stage invention pipeline for training conditional inventors (Figure 5):

回顾一下，出现了一个用于培训有条件的发明者的三阶段发明流水线（图 5）：

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%205.png"></div>

> Figure 5: The Invention Pipeline. (left) A three-staged training pipeline bootstraps from a single example of an invention to an LLM that can output an invention tailored to its current condition. The hope for the future is for such a conditional inventor agent to help seed an open-ended process, wherein interactions between agents and their inventions spur continual innovation. (right) In the Sodarace domain, the conditional inventor observes the terrain, which conditions the LLM to output the speci cation of the desired invention. \
图 5：发明流水线。 （左）一个三阶段的训练流水线，从一个发明的单个示例引导到一个 LLM，该 LLM 可以输出适合其当前条件的发明。 未来的希望是，这样一个有条件的发明者代理人可以帮助播种一个开放式的过程，其中代理人与其发明之间的相互作用会刺激持续的创新。 （右）在 Sodarace 域中，有条件的发明者观察地形，这为 LLM 提供条件以输出所需发明的规范。

1. ELM. Search for a diverse set of example artifacts (e.g. Sodaracers on at ground). \
ELM。 搜索一组不同的示例工件（例如地面上的 Sodaracers）。

2. Pre-train the LLM with examples from ELM. The LLM accordingly learns to output example inventions from the training distribution. \
使用来自 ELM 的示例预训练 LLM。 LLM 相应地学习从训练分布中输出示例发明。

3. Learn to invent conditionally. Splice new conditional inputs onto the LLM and ne tune it through RL to produce appropriate inventions for the conditions it observes. \
学会有条件地发明。 将新的条件输入拼接到 LLM 上，并通过 RL 对其进行微调，以针对其观察到的条件产生适当的发明。

### 4.1 Encoding Sodaracers with Python 通过 Python 编码 Sodaracers

Previous experiments targeting Sodarace have leveraged specialized evolutionary encodings [12]. Instead, in this work plain-text Python code acts as a generic representation for inventions. By showing how Python can be used to represent artifacts in an arbitrary domain, it opens up the possibility of using it as a generic encoding in diverse future domains. More specifically, in the experiments an individual is evaluated by executing its code through the Python interpreter. The product of the interpreter (for a viable individual) is a data structure containing the description of a Sodaracer (i.e. a Python dictionary containing lists of both point masses and springs), which can then be passed to the Sodarace simulator to evaluate the encoded Sodaracer’s behavior. Note that Sodaracers are encoded in Python throughout the invention pipeline, i.e. ELM evolves Python programs and the language models in both latter stages of the pipeline are trained to output Python programs.

以前针对Sodarace的实验已经利用了专门的进化编码[12]。相反，在这项工作中，纯文本的Python代码充当了发明的通用表示。通过展示Python如何被用来表示任意领域的人工制品，它开启了在未来不同领域使用它作为通用编码的可能性。更具体地说，在实验中，一个个体是通过Python解释器执行其代码来进行评估的。解释器的产物（对于一个可行的个体）是一个包含Sodaracer描述的数据结构（即一个包含点质量和弹簧列表的Python字典），然后它可以被传递给Sodarace模拟器来评估编码的Sodaracer的行为。请注意，在整个发明流水线中，Sodaracers是用Python编码的，即ELM演化出Python程序，流水线的后两个阶段的语言模型都被训练成输出Python程序。

Preliminary experiments showed that the diff model’s initial performance (i.e. before fine-tuning) in creating useful perturbations depended upon the design of the "interface" through which Sodaracers were procedurally constructed.

初步实验表明，差异模型在产生有用扰动方面的初始性能（即在微调之前）取决于在程序上构建 Sodaracers 的“界面”的设计。

That is, while a Sodaracer can be constructed in Python by directly adding elements to a Python dictionary with keys such as "joints" and "muscles," a more Pythonic interface (which was more e ective and is what is used in the experiments) is to create a simple class with two methods: "add joint" (to add a spring) and "add muscle" (to add a point mass.) The idea is that such an interface (here encapsulated in a class called "walker creator") is closer to the training distribution of Python code (although still no Sodarace examples in this format exist). For example, below is the encoding of a simple hand-designed square Sodaracer (that is also used in the experiments as a seed), as well as its translation after being executed into a dictionary of joints and muscles. The interface also includes logic for ensuring that the Sodaracer will not break the underlying Box2D physics engine, e.g. that each joint is connected only to so many muscles, that the strength of muscles is limited, and that there is a minimum distance between joints. Note that the benefit of evolving a program that produces a data structure rather than directly evolving the data structure itself relates to the bene ts of indirect encoding (i.e. a program can leverage regularities through loops, conditionals, and functions, to e ciently encode large complex structures) [60]. Figure 6 shows an image of this walker when instan-tiated.

也就是说，虽然 Sodaracer 可以在 Python 中通过使用“关节”和“肌肉”等键直接添加元素到 Python 字典中来构建，但更 Python 化的界面（这更有效，是实验中使用的）就是用两个方法创建一个简单的类：“添加关节”（添加弹簧）和“添加肌肉”（添加点质量）。想法是这样的接口（这里封装在一个名为“walker creator”的类中") 更接近 Python 代码的训练分布（尽管仍然不存在这种格式的 Sodarace 示例）。例如，下面是一个简单的手工设计的方形 Sodaracer（在实验中也用作种子）的编码，以及在执行到关节和肌肉字典后的翻译。该接口还包括确保 Sodaracer 不会破坏底层 Box2D 物理引擎的逻辑，例如每个关节只与这么多肌肉相连，肌肉的力量是有限的，关节之间有最小的距离。请注意，进化产生数据结构的程序而不是直接进化数据结构本身的好处与间接编码的好处有关（即程序可以通过循环、条件和函数利用规律，有效地编码大型复杂结构）[60]。图 6 显示了该 walker 实例化时的图像。

```python
 
 from walk_creator import walker_creator

 def make_square(wc, x0, y0, x1, y1):
 """ Make a square with top left x0,y0 and top right x1,y1 """
 j0 = wc.add_joint(x0, y0)
 j1 = wc.add_joint(x0, y1)
 j2 = wc.add_joint(x1, y1)
 j3 = wc.add_joint(x1, y0)

 return j0, j1, j2, j3

 def make_walker():
 wc = walker_creator()

 # the main body is a square

 sides = make_square(wc, 0, 0, 10, 10)

 center = wc.add_joint(5, 5)

 # connect the square with distance muscles 
 for k in range(len(sides)-1):
  wc.add_muscle(sides[k], sides[k+1]) 
 wc.add_muscle(sides[3], sides[0])

 # one prong of the square is a distance muscle 
 wc.add_muscle(sides[3], center)

 # the other prongs from the center of the square are active 
 wc.add_muscle(sides[0], center, False, 5.0, 0.0) 
 wc.add_muscle(sides[1], center, False, 10.0, 0.0) 
 wc.add_muscle(sides[2], center, False, 2.0, 0.0)

return wc.get_walker()

```

>Listing 1: Example Sodaracer-generating program. \
清单1：Sodaracer生成程序的例子。

## 5 Pipeline Stage 1:  Data Generation through ELM 流水线阶段1：通过ELM生成数据

Recall that the aim in this rst stage is to generate a large variety of high-quality examples from a single example starter program through ELM. In this stage of the pipeline, the Sodarace environment is a simple at terrain.

回顾一下，在这个第一阶段的目的是通过ELM从一个单一的例子启动程序中产生大量的高质量例子。在流水线的这个阶段，Sodarace环境是一个简单的地形。

Recall that ELM in this experiment will evolve through MAP-Elites (Figure 3) [51]. The core of MAP-Elites is a uniformly-spaced grid of niches (called the map), that spans user-speci ed dimensions of solution diversity, called the behavior characterization. In experiments here, the behavior characterization consists of the height, width, and mass of Sodaracers, and the map is a 12 &times; 12 &times; 12 grid into which any behavioral characterization can be mapped. Upon initialization, a single hand-designed solution is evaluated and placed into the map. In each iteration thereafter, an inhabited niche is randomly chosen and the solution within that niche is perturbed by the diff model and evaluated. The new candidate solution is assigned its niche from its behavior characterization, and if that niche is unfilled or the new solution outperforms the niche’s current inhabitant, it becomes the champion of that niche; otherwise, the candidate is discarded. In this way, over iterations of search, the map gradually lls with increasingly diverse and high-quality solutions.

回想一下，本实验中的 ELM 将通过 MAP-Elites 进化（图 3）[51]。 MAP-Elite 的核心是一个均匀间隔的生态位网格（称为地图），它跨越了用户指定的解决方案多样性维度，称为行为表征。在这里的实验中，行为特征包括 Sodaracers 的高度、宽度和质量，并且该地图是一个 12 &times; 12 &times; 12 网格，可以将任何行为特征映射到其中。初始化后，将评估单个手工设计的解决方案并将其放入地图中。在此后的每次迭代中，随机选择一个有人居住的生态位，并且该生态位内的解决方案受到差异模型的干扰并进行评估。新的候选解决方案根据其行为特征分配其生态位，如果该生态位未填充或新解决方案优于该生态位的当前居民，它将成为该生态位的冠军；否则，候选人被丢弃。这样，经过反复的搜索，地图逐渐填满了越来越多样化和高质量的解决方案。

```json
{
 "joints": [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5)],
 "muscles": [
  [0, 1, -"type": "distance"}],
  [1, 2, -"type": "distance"}],
  [2, 3, -"type": "distance"}],
  [3, 0, -"type": "distance"}],
  [3, 4, -"type": "distance"}],
  [0, 4, -"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
  [1, 4, -"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
  [2, 4, -"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
 ],
}
```

> Listing 2: Intermediate Sodaracer representation from running the above Python seed program. \
清单 2：运行上述 Python 种子程序的中间 Sodaracer 表示。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%206.png"></div>

>Figure 6: Instantiation of a hand-designed square Sodaracer. A video of this walker can be viewed at https://y2u.be/jeP8Nsulu48 \
图 6：手工设计的方形 Sodaracer 的实例化。 可以在 https://y2u.be/jeP8Nsulu48 观看此步行者的视频

To address the need for seed solutions, four simple seeds were written that explore different architectural motifs: the Square seed, the Radial seed, and two CPPN-like seeds (CPPN stands for compositional pattern-producing network [61]); note that source code for these seeds is provided in Appendix B. The Square seed instantiates a polygon-like bias, by including a function that creates a square composed of four masses from two coordinates, and code that calls that function and connects the masses together with a for-loop. The Radial seed instead implements a radial bias by replacing the square-generating function with a function that places a given number of masses in a circular shape. Finally, the CPPN-like seeds roughly implement the CPPN-based encoding used by previous work in Sodarace [12], i.e. it places masses and connects springs between them as a mathematical function of their coordinates. The CPPN-based seed’s code can be neatly divided into (1) implementing the core functionality of a CPPN, and (2) describing a particular instantiation of a CPPN, and thus enables exploring the consequences of letting core functionality of the encoding itself evolve. To this end, there are two CPPN seeds, one in which the CPPN encoding is fixed, called the CPPN-Fixed seed, and one where it is mutable, called the CPPN-Mutable seed. Note that these seed programs were not highly-tuned as the videos in Figure 7 highlight.

为了满足种子解决方案的需求，编写了四个简单的种子来探索不同的架构主题：方形种子、径向种子和两个类似 CPPN 的种子（CPPN 代表组合模式生成网络 [61]）；请注意，附录B中提供了这些种子的源代码。Square 种子实例化了类似多边形的偏差，方法是包含一个函数，该函数从两个坐标创建一个由四个质量组成的正方形，以及调用该函数并将质量连接在一起的代码带有for循环。径向种子通过将平方生成函数替换为将给定数量的质量置于圆形形状的函数来实现径向偏差。最后，类似 CPPN 的种子大致实现了 Sodarace [12] 中先前工作使用的基于 CPPN 的编码，即它放置质量并将弹簧连接在它们之间，作为它们坐标的数学函数。基于 CPPN 的种子代码可以巧妙地分为 (1) 实现 CPPN 的核心功能，以及 (2) 描述 CPPN 的特定实例化，从而能够探索让编码本身的核心功能发展的结果。为此，有两种 CPPN 种子，一种是 CPPN 编码是固定的，称为 CPPN-Fixed 种子，另一种是可变的，称为 CPPN-Mutable 种子。请注意，这些种子程序没有经过高度调整，如图 7 中的视频突出显示的那样。

<div align=center><img width = '' height ='350' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%207.png"></div>

> Figure 7: The three seed solutions. From top to bottom: CPPN seed, radial seed, and square seed. A video of these walkers is at https://y2u.be/jeP8Nsulu48 (same video as for Figure 6). \
图 7：三种种子解决方案。 从上到下：CPPN 种子、径向种子和方形种子。 这些步行者的视频位于 https://y2u.be/jeP8Nsulu48（与图 6 相同的视频）。

### 5.1 Experimental Details and Results 实验细节和结果

Three independent runs of ELM were conducted with each seed, running for 1,024,000 evaluations each (composed of 2,000 iterations of 512 diffs per iteration). A 300M parameter pretrained diff model [31] served as the perturbation operator in these experiments.

对每个种子进行了三次独立的 ELM 运行，每次运行 1,024,000 次评估（由每次迭代 512 差异的 2,000 次迭代组成）。 在这些实验中，一个 300M 参数的预训练差异模型 [31] 作为扰动算子

One metric of success for ELM is the number of niches filled, which represents the diversity of data generated by ELM, under the hypothesis that such diverse data will benefit later pipeline stages. Figure 8 shows that runs of ELM tend to discover a large proportion of niches, highlighting how the system can bootstrap from a single user-provided example to fill the space of desired possibilities. However, the speed of spreading through niches varies across seeds; in particular, introducing loops and/or function composition is required for the Square seed to spread into high-mass niches (e.g. to connect many squares together), which emerges slowly in some runs.
 
ELM 成功的一个衡量标准是填充的生态位数量，它代表 ELM 生成的数据的多样性，假设这种多样化的数据将有利于后期流水线阶段。 图 8 显示 ELM 的运行倾向于发现大比例的生态位，突出显示了系统如何从单个用户提供的示例引导以填充所需的可能性空间。 然而，通过生态位传播的速度因种子而异； 特别是，Square 种子需要引入循环和/或函数组合才能传播到高质量的生态位（例如将许多正方形连接在一起），这在某些运行中会缓慢出现。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%208.png"></div>

> Figure 8: Amount of niches filled across seeds. The figure shows the percentage of all niches (1,728 in total) that are filled by the end of ELM runs across different seeds. Results are averaged across three independent runs for each seed. In general, nearly all seeds fill the map, although the Square seed proceeds more slowly than other seeds. \
图8：不同种子的生态位填充量。该图显示了在不同种子的ELM运行结束时，所有生态位（共1728个）被填充的百分比。结果是每个种子三次独立运行的平均数。一般来说，几乎所有的种子都填满了地图，尽管广场种子比其他种子进行得更慢。

Beyond diversity, the quality of solutions is also important. A gross measure of quality is the maximum fitness discovered by runs, shown in Figure 9. A more nuanced metric that takes both quality and diversity into account is the QD score [50], calculated as the sum of the performance of all champions in the final map. This metric, shown averaged over runs in Figure 10, rewards both quality (having higher scores in each niche) and diversity (having discovered more niches), and thus serves as a succinct measure of ELM’s goal of accumu-lating diverse, high-quality solutions (and in later stages in the pipeline, of how well an LLM has modeled the distribution of solutions that ELM has uncov-ered). Attainment of QD di ers across seeds; while the CPPN seed uncovers diversity most quickly, the Radial seed generates higher-quality solutions on average. The relationship between the seed and the products of search is complex and deserves further future study (see also Appendix D for further analysis of seed robustness).

除了多样性，解决方案的质量也很重要。衡量质量的一个粗略标准是运行发现的最大适配度，如图9所示。一个考虑到质量和多样性的更细微的指标是QD得分[50]，计算为最终地图中所有冠军的表现之和。这个指标在图10中显示了运行的平均值，它既奖励质量（在每个生态位中拥有更高的分数），也奖励多样性（发现了更多的生态位），因此可以作为ELM积累多样化、高质量解决方案的目标的一个简洁的衡量标准（在流水线的后期阶段，可以衡量LLM对ELM所发现的解决方案的分布的建模程度）。不同种子的QD实现情况不同；CPPN种子发现多样性的速度最快，而Radial种子平均产生的解决方案质量更高。种子和搜索结果之间的关系很复杂，值得今后进一步研究（关于种子鲁棒性的进一步分析也见附录D）。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%209.png"></div>

> Figure 9: Maximum fitness across seeds. The maximum performance attained on average by different seeds is shown. The results suggest that ELM’s capacity to find hightness solutions is somewhat robust to seed design. \
图9：不同种子的最大适应性。图中显示了不同种子平均达到的最大性能。结果表明，ELM寻找高度解决方案的能力对种子设计有一定的稳健性。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2010.png"></div>

> Figure 10: Quality diversity score across seeds. Shown is the average nal QD score attained by runs initialized from di erent seeds. The conclusion is that ne-tuning the di model has a significant impact on attained QD score, as does the choice of seed. \
图10：不同种子的质量多样性得分。显示的是由不同种子初始化的运行所获得的平均QD分数。结论是，对模型进行微调对获得的QD分数有很大影响，种子的选择也是如此。

Fine-tuning the diff model on accepted diffs from an initial series of runs greatly increased performance (Figure 11); while Sodarace-generating programs are out-of-distribution for the pretrained diff model (applying a Python encoding to this domain is a novel enterprise), fine-tuning effectively aligns the diff model with the domain, an interesting result. Figure 11c shows how the fine-tuned diff model produces a significantly higher percentage of diffs that are valid (i.e. able to be applied) and runnable (i.e. the patched program is executable). Because of their higher performance, the output of runs applying the fine-tuned diff model are the ones passed to later stages in the pipeline.

在最初的一系列运行中对接受的差异进行微调，大大提高了性能（图11）；虽然Sodarace生成的程序对预训练的差异模型来说是不适用的（在这个领域应用Python编码是一项新的事业），但微调有效地使差异模型与该领域保持一致，这是一个有趣的结果。图11c显示了微调后的差异模型产生的有效（即能够被应用）和可运行（即修补后的程序可执行）的差异比例明显提高。由于其更高的性能，应用微调差异模型的运行输出被传递到流水线的后期阶段。

<div align=center><img width = '' height ='600' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2011.png"></div>

> Figure 11: The impact of fine-tuning the diff model on the performance of ELM. For both the pretrained diff model and the fine-tuned one, shown are (a) the number of niches reached, (b) QD score of the produced map, and (c) percentage of valid/runnable diffs proposed. The experiments demonstrate that fine-tuning the diff model improves performance of the evolutionary process across all three metrics. \
图 11：微调差异模型对 ELM 性能的影响。 对于预训练的差异模型和微调模型，显示的是 (a) 达到的生态位数量，(b) 生成的地图的 QD 分数，以及 (c) 提出的有效/可运行差异的百分比。 实验表明，微调差异模型可以提高所有三个指标的进化过程的性能。


Note that further rounds of ne-tuning are possible (e.g. fine-tuning the diff model again from the improved products of the first round); however preliminary experiments showed diminishing returns. Future work could explore how to continually improve such models, such as by identifying and encouraging more impactful perturbations instead of including and weighting equally all accepted diffs.

请注意，可以进行进一步的微调（例如，从第一轮的改进产品再次微调差异模型）； 然而，初步实验显示收益递减。 未来的工作可以探索如何不断改进此类模型，例如通过识别和鼓励更具影响力的扰动，而不是平等地包括和加权所有已接受的差异。

The seeds and fine-tuned diff model also qualitatively impact the kinds of solutions discovered by ELM. While the Radial seed performs well quantitatively (in terms of quality and diversity), it turns out that its products tend to exploit chaotic dynamics that seem overfit to the flat terrain (this hypothesis is tentatively validated in the Stage 3 experiments). The Square and CPPN seeds in contrast are more likely to output inventions that leverage more predictable dynamics. For these reasons, the Radial seed runs were not ultimately used in future stages.

种子和微调差异模型也定性地影响 ELM 发现的解决方案的种类。 虽然径向种子在数量上表现良好（在质量和多样性方面），但事实证明，它的产品倾向于利用似乎过度适合平坦地形的混沌动力学（这一假设在第 3 阶段的实验中得到初步验证）。 相比之下，Square 和 CPPN 种子更有可能产出利用更可预测动态的发明。 由于这些原因，径向种子运行最终并未在未来阶段使用。

A video selection of the highest-quality Sodaracers from these initial runs that showcases the considerable diversity uncovered can be viewed at https://y2u.be/QNyNtvwA9FI. An example of a lineage of Sodaracers progressing from the Square seed to a high-quality final Sodaracer can be seen at https://y2u.be/M9pAJuX6dyM. In short, ELM shows that by combining the an intelligent LLM-based mutation operator with a QD algorithm it is possible to generate hundreds of thousands of working training examples in a completely novel domain where no data was previously available.

可以在 https://y2u.be/QNyNtvwA9FI 上观看从这些初始运行中挑选出的最高质量 Sodaracers 的视频，这些视频展示了所发现的相当大的多样性。 可以在 https://y2u.be/M9pAJuX6dyM 中查看从 Square 种子发展为高质量最终 Sodaracer 的 Sodaracer 谱系示例。 简而言之，ELM 表明，通过将基于 LLM 的智能变异算子与 QD 算法相结合，可以在以前没有可用数据的全新领域中生成数十万个工作训练示例。

## 6 Pipeline Stage 2: Language Model Training 流水线阶段 2：语言模型训练

The product of Stage 1 is a collection of programs, whereas Stage 3 RL requires an initial model that can output valid Sodaracer-generating programs. Thus, the second stage of the invention pipeline fine-tunes an LLM on the products of ELM, which serves as the initialization for an RL-based conditional inventor.To do so first requires compiling the results of Stage 1 into a fine-tuning dataset.

第一阶段的产物是一个程序的集合，而第三阶段RL需要一个能够输出有效的Sodaracer生成程序的初始模型。因此，发明流水线的第二阶段对ELM产品的LLM进行微调，作为基于RL的条件发明者的初始化。要做到这一点，首先需要将第一阶段的结果编译成一个微调数据集。

While there are many ways to distill a dataset of programs from runs of ELM, a simple thresholded approach is adopted here (although see Appendix E for another simple approach that did not work in practice). The main idea is to append all reasonably-capable solutions for each niche.

虽然有很多方法可以从ELM的运行中提炼出程序的数据集，但这里采用的是一种简单的阈值方法（不过，另一种简单的方法在实践中并不奏效，见附录E）。主要的想法是为每个生态位附加所有合理可行的解决方案。

In more detail, from each run all solutions ever admitted to the map are included, subject to meeting a minimal bar for performance. Some parts of the behavior space offer more stringent challenges (i.e. it is more diffcult to locomote when required to be tall but not wide and to have low mass), and yet in some terrains encountered in Stage 3, those kinds of solutions may yet be most effective despite their low level of absolute performance. Thus, for each niche, the maximum performance across all runs is calculated, and the minimal bar for inclusion is set as a percentage of that per-niche score. With a higher percentage threshold, less data is included, but the quality of that data will be higher.

更详细地说，在满足最低性能标准的前提下，每一次运行中所有被允许进入地图的解决方案都包括在内。行为空间的某些部分提供了更严格的挑战（例如，当要求高而不宽和低质量时，运动更困难），然而在第三阶段遇到的一些地形中，这些类型的解决方案可能是最有效的，尽管其绝对性能水平低。因此，对于每个生态位，计算出所有运行中的最大性能，并将列入的最低标准设定为每个生态位得分的百分比。随着百分比阈值的提高，纳入的数据就会减少，但数据的质量也会提高。

As noted in the previous section, solutions from the Radial seed were qualitatively chaotic. Furthermore, preliminary experiments suggest that such chaotic behavior significantly harms downstream Stage 3 performance. For these reasons Radial runs of ELM were excluded from the LLM datasets. Datasets for each of the remaining treatments were compiled from 9 runs from ELM with the fine-tuned diff model (3 runs for each of the Square, CPPN-Fixed, and CPPN-Mutable seeds). In total, the 50% cut-off threshold dataset consisted of 280K examples, and the 80% cut-off threshold dataset contained a subset of 95K of those examples.

如上一节所述，径向种子的解决方案在质量上是混乱的。 此外，初步实验表明，这种混乱行为严重损害了下游阶段 3 的性能。 由于这些原因，ELM 的径向运行被排除在 LLM 数据集中。 其余每个处理的数据集是从 ELM 的 9 次运行中编译的，使用微调的差异模型（Square、CPPN-Fixed 和 CPPN-Mutable 种子各运行 3 次）。 总体而言，50% 截止阈值数据集包含 280K 示例，80% 截止阈值数据集包含 95K 这些示例的子集。

A variety of pretrained code-generating models were then fine-tuned with these examples (using the standard LLM log-probability loss), leaving out 5% of the data to serve as a test set. Models ranging from 0.1M to 680M parameters were trained (architectural details for these models can be seen in Appendix C). Also, as a control to support the hypothesis that Sodarace models benefit from code-generation pretraining, a 300M model was also trained instead from a random initialization (signified with "RI" in charts that follow).


然后用这些例子对各种预训练的代码生成模型进行微调（使用标准的LLM对数概率损失），留下5%的数据作为测试集。训练了从0.1M到6.8M参数的模型（这些模型的结构细节可以在附录C中看到）。另外，作为支持Sodarace模型受益于代码生成预训练这一假设的对照，一个300M的模型也从随机初始化中训练出来（在后面的图表中用 "RI "表示）。

Minimum test-losses (i.e. loss on generated Sodaracers held-out from the fine-tuning dataset) of the 80% Percentage Threshold models are shown in Figure 12. The 50% Percentage Threshold models exhibit qualitatively similar results across model size (but as both thresholds represent different datasets, loss values are not directly comparable between them). The conclusions are that model sizes above 85M may not better fit the data, and that random initialization does hurt performance relative to fine-tuning from a model pretrained on code.

图 12 显示了 80% 百分比阈值模型的最小测试损失（即生成的 Sodaracers 的损失从微调数据集中保留）如图 12 所示。50% 百分比阈值模型在模型大小上表现出质量相似的结果（但两者 阈值代表不同的数据集，损失值在它们之间不能直接比较）。 结论是，超过 85M 的模型大小可能无法更好地拟合数据，并且相对于在代码上预训练的模型进行微调，随机初始化确实会损害性能。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2012.png"></div>

> Figure 12: Test loss across model sizes. The minimum test loss achieved by training runs on the 80% Percentage Threshold dataset across model sizes is shown. Model sizes above 85M may not better-fit the data, and random initialization hurts performance. \
图12：不同模型规模的测试损失。图中显示了在80%的百分比阈值数据集上训练运行时，获得了在不同模型规模上最小测试损失。超过85M的模型规模可能无法更好地适应数据，而随机初始化会损害性能。

However, loss is not the whole story. The interesting question for Stage 2 is whether the LLMs trained from the data generated in Stage 1 can generate the same diversity and quality of data. Therefore, the QD score metric and number of niches discovered (both of which were also reported for Stage 1) are calculated for samples taken from trained LLMs. Because these metrics can be maximized by a model that memorizes the data, and because empirically QD score was more correlated with loss on the training set rather than the test set, the LLM checkpoint for each model is selected on the basis of lowest training loss. In particular, 1,024 samples are taken from each model, which are then evaluated and inserted into a new MAP-Elites map. For comparison, the same metrics are calculated using the Stage 1 dataset, by taking the same number of samples from it and evaluating them in the same way. These results are shown in Figure 13, highlighting that the model samples achieve a similar level of performance as dataset samples, suggesting that they have modeled the data well. Also, there is a slight but consistent QD bene t from models trained on the 80% cuto dataset, reflecting the higher average QD of that dataset.

然而，损失并不是全部。第 2 阶段的有趣问题是，根据第 1 阶段生成的数据训练的 LLM 是否可以生成相同的数据多样性和质量。因此，从训练有素的 LLM 中提取的样本计算了 QD 得分指标和发现的生态位数量（这两者也都被用于了第 1 阶段）。因为这些指标可以通过一个记忆数据的模型来最大化，并且因为经验上 QD 分数与训练集上的损失而不是测试集上的相关性更高，所以每个模型的 LLM 检查点是根据最低的训练损失来选择的。特别是，从每个模型中抽取 1,024 个样本，然后对其进行评估并插入到新的 MAP-Elite 地图中。为了比较，使用第 1 阶段数据集计算相同的指标，方法是从中获取相同数量的样本并以相同的方式评估它们。这些结果如图 13 所示，突出显示模型样本实现了与数据集样本相似的性能水平，表明它们已经很好地对数据进行了建模。此外，在 80% cuto 数据集上训练的模型有轻微但一致的 QD 优势，反映了该数据集更高的平均 QD。

<div align=center><img width = '' height ='500' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2013.png"></div>

> Figure 13: Measuring the quality and diversity of model samples. Two metrics evaluating samples from trained LLMs are shown (across model size and training dataset): (a) the percentage of niches discovered and (b) the QD score achieved. The 80% threshold dataset is on average less diverse but of higher quality than the 50% threshold dataset, and induces the same properties in models trained upon it. There is not a trend in increasing quality or diversity as model size increases beyond 85M, and random initialization hurts performance. \
图 13：衡量模型样本的质量和多样性。 显示了评估来自受过训练的 LLM 的样本的两个指标（跨模型大小和训练数据集）：（a）发现的生态位百分比和（b）达到的 QD 分数。 与 50% 阈值数据集相比，80% 阈值数据集的平均多样性较低，但质量更高，并且在基于其训练的模型中具有相同的属性。 随着模型大小增加到超过 85M，并且随机初始化会损害性能，因此没有增加质量或多样性的趋势。

A natural further question is how well the model will do when taken out of distribution, i.e. how well has it really internalized the dynamics of Sodarace? That is, the training and test set for fine-tuning are taken from the same runs, and thus the model will likely have encountered all of the motifs in the test set, and so it may not be a representative test of how well the model will generalize in the future. A preliminary test in this spirit is to take the first half of the Python programs describing several inventions from unseen runs, and explore the capacity of different models to generate functional completions. Though the Radial seed usually produced chaotic Sodaracers, in one preliminary run of ELM with the Radial seed, a functional wheel was discovered. As noted previously data from this run (or any other radial runs) was not used to train the models in Stage 2, nor was it used to fine-tune the diff model in Stage 1; thus the ability to complete the wheel can serve as a proxy for generalization. Similarly, two other high-performing individuals were taken from other preliminary runs of the CPPN seed and the Square seed, to create a set of three out-of-distribution completion tests. See Figure 14 for visualizations of these walkers, including videos; source code for these generalization examples can be found in Appendix F). Note that further tests of generalization are documented in Appendix H.

<!-- TODO function wheel-->
一个自然的进一步问题是，当模型从分布中取出时，它的表现如何，也就是说，它真正内化了Sodarace的动态的程度如何？也就是说，用于微调的训练集和测试集取自相同的运行，因此模型很可能已经遇到了测试集中的所有图案，所以它可能不是对模型在未来的泛化程度的一个代表性测试。本着这种精神，一个初步的测试是把描述几个发明的Python程序的前半部分从未见过的运行中拿出来，探索不同模型产生功能完成的能力。尽管Radial种子通常会产生混乱的Sodaracers，但在一次使用Radial种子的ELM初步运行中，发现了一个功能轮。如前所述，这次运行（或任何其他径向运行）的数据没有被用来训练第二阶段的模型，也没有被用来微调第一阶段的差异模型；因此，完成Wheel的能力可以作为泛化的一个代理。同样，从CPPN种子和Square种子的其他初步运行中抽取了另外两个表现出色的个体，以创建一组三个分布外的完成测试。这些walker的可视化情况见图14，包括视频；这些泛化例子的源代码可以在附录F中找到）。请注意，附录H中记录了进一步的泛化测试。

<div align=center><img width = '' height ='300' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2014.png"></div>

> Figure 14: Generalization tests. In this test, the model is asked to complete samples, taken from the rst half of the dataset from the unseen runs. These unseen originals are shown in the videos at https://y2u.be/8C2K5fk28HI. From top to bottom: Wheel, from radial seed; Galloper, from square seed; Runner, from CPPN seed. \
图14：概括性测试。在这个测试中，模型被要求完成样本，取自未见过的数据集的前一半。这些未见过的原件在视频中显示，https://y2u.be/8C2K5fk28HI。从上到下。Wheel，来自径向种子；Galloper，来自方形种子；Runner，来自CPPN种子。

For each of the three completion tasks, 1,024 completion samples are taken from each model and then evaluated in simulation. In contrast to the in-distribution metrics, in this generalization-focused test, performance was more correlated with the model’s test loss rather than training loss, and thus what checkpoint to evaluate for each model was selected on the basis of lowest test loss. Results are shown in Figure 15, highlighting that larger models, and those trained on the 80% threshold, generally perform better at this task. Note that the randomly-initialized (RI) 300M model significantly underperforms, providing more evidence that pretraining on code provides a valuable prior.

对于三个完成任务中的每一个，从每个模型中抽取1,024个完成样本，然后在模拟中进行评估。与分布内指标相比，在这个以泛化为重点的测试中，性能与模型的测试损失而不是训练损失更相关，因此对每个模型评估什么检查点是根据最低测试损失来选择的。结果显示在图15中，突出了较大的模型，以及那些在80%阈值上训练的模型，通常在这个任务中表现更好。请注意，随机初始化（RI）的300M模型明显表现不佳，提供了更多的证据，表明对代码的预训练提供了一个有价值的先验。

<div align=center><img width = '' height ='200' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2015.png"></div>


> Figure 15: Out of distribution completion performance. Shown is the percentage of the original solutions’ performance that is attained by completions from trained LLMs. The percentage shown is the maximum attained over 1,024 independent completion samples from each model. The results are averaged over three out-of-distribution solutions (taken from runs not included in LLM training). The conclusion is that the 80% threshold models perform better than the 50% threshold, and that there is no obvious trend in performance once model size reaches 85M parameters. \
图15：超出分布的完成性能。所显示的是经过训练的LLM的完成度所达到的原始解决方案的性能百分比。所显示的百分比是每个模型的1024个独立完成样本所达到的最大值。这些结果是三个非分布式解决方案的平均值（取自不包括在LLM训练中的运行）。结论是80%的阈值模型比50%的阈值表现更好，而且一旦模型大小达到85M的参数，性能就没有明显的趋势。

Videos of the best-performing sample for the Wheel completion from each model are at https://y2u.be/-LW2cCwSdRU (for the 80% threshold dataset; the random-initialized 300M model is not shown because it generated no valid samples for this completion). For the Galloper and Runner completions, the structure and/or behavior of completions often does not match the original sample (especially for the Galloper). In the following linked video, a higher-performing completion is shown for both of the Galloper and the Runner: https: //y2u.be/XR3L4cZ83xU.

每个模型的Wheel完成的最佳表现样本的视频在https://y2u.be/-LW2cCwSdRU（对于80%的阈值数据集；随机初始化的300M模型没有显示，因为它对这个完成没有产生有效样本）。对于Galloper和Runner完成度，完成度的结构和/或行为往往与原始样本不一致（尤其是Galloper）。在下面的链接视频中，Galloper和Runner都显示了一个性能更高的完成度：https: //y2u.be/XR3L4cZ83xU。

Overall, these results show that an LLM can effectively integrate synthetic data generated through ELM in a novel domain.

总的来说，这些结果表明，在一个新的领域，LLM可以有效地整合通过ELM产生的合成数据。

## 7 Pipeline Stage 3: Conditional RL 流水线阶段3：有条件RL

In the final stage, reinforcement learning (RL) is invoked to fine-tune the pretrained LLM output by Stage 2 of the pipeline. The goal is to produce a model that outputs Python programs representing Sodaracers in response to particular terrains. Importantly, the output of Stage 2 is an unconditional model, in the sense that it samples Sodaracers from a distribution de ned by the output of Stage 1, without considering the terrain in which the samples will be deployed. The first step in Stage 3 is thus to convert the model to a conditional one, i.e. a model that accepts terrains as inputs, and produces samples of Sodaracers in response.

在最后阶段，强化学习（RL）被调用来微调流水线第二阶段预训练的LLM输出。目标是产生一个模型，输出代表Sodaracers的Python程序来应对特定的地形。重要的是，第二阶段的输出是一个无条件的模型，在这个意义上，它从第一阶段的输出所确定的分布中对Sodaracers进行采样，而不考虑采样将被部署在什么地形。因此，第三阶段的第一步是将该模型转换为有条件的模型，即接受地形作为输入，并产生Sodaracers的样本作为回应。

To achieve this functional form, we rst introduce the notion of a terrain embedding network (TEN). The role of the TEN is to map a representation of the terrain to a representation that can be used by the model to sample conditionally. In particular, the output of TENs is a vector (or sequence of vectors) in <font face="Times New Roman"><b> <i>d</i></b></font>, the dimension in which the model embeds tokens. That way, the output of the TEN can be treated as the activation from a given prefix, and the model can proceed in effect now sampling conditioned on the output of the TEN.

为了实现这种功能形式，我们首先引入了地形嵌入网络（TEN）的概念。TEN的作用是将地形的表征映射到模型可用于有条件采样的表征。特别是，TEN的输出是 <font face="Times New Roman"><b> <i>d</i></b></font> 中的一个向量（或向量序列），即模型嵌入令牌的维度。这样，TEN的输出可以被视为来自给定前缀的激活，而模型实际上可以进行以TEN的输出为条件的采样。

Concretely, an unconditional autoregressive LLM de nes a sampling distribution over a sequence of tokens <font face="Times New Roman"><b><i>x</i> = ( <i>x<sub>1</sub>, ... ,x<sub>n</sub></i></b> )</font> as <img height ='20' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/formula%201.png"/>
In this stage, we introduce the additional module <font face="Times New Roman"> <i>f</i> <sub>TEN</sub> </font>, which represents terrains <font face="Times New Roman"><b><i>t</i></b></font> in <font face="Times New Roman">&Ropf;<b><sup><i>d</i></b></sup></font>. As <font face="Times New Roman"> <i>f</i> <sub>TEN</sub> (x) &isin; &Ropf;<b><sup><i>d</i></b></sup></font>, we can consider the resulting conditional model without further modi cation:

具体来说，无条件自回归 LLM 将标记序列 <font face="Times New Roman"><b><i>x</i> = ( <i>x<sub>1</sub>, ... ,x<sub>n</sub></i></b> )</font>上的抽样分布定义为 <img height ='20' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/formula%201.png"/>。 在这个阶段，我们引入了额外的模块 <font face="Times New Roman"> <i>f</i> <sub>TEN</sub> </font>，它表示<font face="Times New Roman">&Ropf;<b><sup><i>d</i></b></sup></font> 中的地形 <font face="Times New Roman"><b><i>t</i></b></font>。 作为 <font face="Times New Roman"> <i>f</i> <sub>TEN</sub> (x) &isin; &Ropf;<b><sup><i>d</i></b></sup></font>，我们可以考虑得到的条件模型，无需进一步修改：

<div align=center><img width = '' height ='30' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/equal%202.png"></div>

This approach is similar to the controllable transformer proposed by Keskar et al. [65], but with the conditional codes being the output of a TEN, rather than particular tokens from the existing vocabulary.

这种方法类似于Keskar等人提出的可控转化器[65]，但条件代码是TEN的输出，而不是现有词汇中的特定标记。

Given a distribution over terrains <font face="Times New Roman"><b><i>p(t)</i></b></font>, an RL setting is constructed to train the parameters of the TEN and further netune the LLM parameters to the conditional setting. In particular, an episode now consists of sampling <font face="Times New Roman"><b><i>t ~ p(t)</i></b></font>, and sampling a program from the conditional distribution de ned in Equation (1). The program is converted to a Sodaracer, evaluated in simulation with the terrain <font face="Times New Roman"><b><i>t</i></b></font>, and the reward is defined as the absolute distance traversed by the Sodaracer in a given period of time.

鉴于地形分布 <font face="Times New Roman"><b><i>p(t)</i></b></font> ，构建了一个RL设置来训练TEN的参数，并进一步将LLM的参数调整到条件设置。特别是，现在的阶段包括对 <font face="Times New Roman"><b><i>t ~ p(t)</i></b></font> 进行采样，并从方程（1）中定义的条件分布中采样一个程序。该程序被转换为Sodaracer，在模拟地形 <font face="Times New Roman"><b><i>t</i></b></font> 的情况下进行评估，奖励被定义为Sodaracer在一定时间内穿越的绝对距离。

### 7.1 Terrain Distributions 地形分布

In this experiment, the distribution over terrains that the model is exposed to is chosen to explore the viability of producing conditional inventors with the Invention Pipeline. The future vision is to lay the groundwork for the ability to deploy agents capable of conditional invention in rich, potentially multi-agent environments that support the development of open-ended processes. In such settings, it stands to reason that learning to output complex artifacts conditioned on observations of the environment would be a prerequisite to ongoing open-ended innovation.

在这个实验中，选择模型所接触到的地形上的分布来探索用发明流水线产生有条件的发明者的可行性。未来的愿景是为在丰富的、潜在的多代理环境中部署能够进行有条件发明的代理打下基础，这些环境支持开放式进程的发展。在这样的环境中，学习根据对环境的观察输出复杂的人工制品是持续的开放式创新的先决条件，这是合理的。

However, in preliminary experiments in the Sodarace domain, learning tended to "gravitate" towards collapsed solutions, wherein a single program is produced that achieves reasonable performance on a subset of the terrains in the distribution support. To reduce the viability of such an outcome and simulate a scenario where conditionality is essential, a small and discrete set of terrains for which a single program cannot achieve good performance provides a test where conditional solutions should be significantly more advantageous.

然而，在Sodarace领域的初步实验中，学习倾向于 "吸引" 塌陷的解决方案，即产生一个单一的程序，在分布支持中的一个地形子集上实现合理的性能。为了降低这种结果的可行性并模拟条件至关重要的场景，单个程序无法实现良好性能的一组小而离散的地形提供了一个测试，其中条件解决方案应该更为有利。

In the experiments, uniform distributions are considered over sets of terrains as illustrated in Figure 16. Two subsets are considered, both of which contain left-wall and right-wall. One set additionally contains tunnel, and the other includes bumpy. These sets were speci cally chosen such that the models are incapable of producing a single Sodaracer that achieves good performance on all terrains; to maximize the learning objective, the model must leverage the TEN to incorporate conditionality.

在实验中，如图 16 所示，考虑了地形集上的均匀分布。考虑了两个子集，它们都包含左墙和右墙。 一组额外包含隧道，另一组包含颠簸。 这些套装经过特别选择，因此这些模型无法生产出在所有地形上都能获得良好性能的单一 Sodaracer； 为了最大化学习目标，模型必须利用 TEN 来合并条件。

<div align=center><img width = '' height ='300' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2016.png"></div>

> Figure 16: Terrains used in experiments. A small set of terrains from which distributions that force conditionality can be constructed. The terrains are (a) left-wall, (b) right-wall, (c) bumpy, and (d) tunnel. The Sodaracers produced by the models are incapable of scaling the walls in left-wall and right-wall, and therefore must produce di erent Sodaracers for these two terrains. Similarly, achieving good performance in the tunnel terrain can only be achieved with short Sodaracers, which struggle to locomote as quickly as taller ones, encouraging the model to distinguish between these terrains. Finally, Sodaracers that are pro cient in locomotion on at terrains tend to perform poorly on bumpy, encouraging the model to produce yet another Sodaracer for this terrain. In contrast to tunnel, which requires Sodaracers with a particular morphology, achieving good performance on bumpy requires modifying the way Sodaracers locomote. Example Sodaracers are added to the gures to illustrate the scale of the terrains. \
图16：实验中使用的地形。一组小的地形，从中可以构建出强制条件的分布。这些地形是（a）左墙，（b）右墙，（c）颠簸，和（d）隧道。由模型产生的Sodaracers无法适应左墙和右墙的墙体，因此必须为这两种地形产生不同的Sodaracers。同样，在隧道地形中实现良好的性能只能用矮小的Sodaracers，它们很难像高大的Sodaracers那样快速移动，这就鼓励了模型对这些地形进行区分。最后，在地形上有利于运动的Sodaracers往往在颠簸的地形上表现不佳，鼓励模型为这种地形生产另一种Sodaracer。与需要具有特定形态的Sodaracers的隧道相比，要想在颠簸路面上取得良好的性能，需要修改Sodaracers的运动方式。图中加入了Sodaracers的例子，以说明地形的规模。

### 7.2 Parametrizing TENs 参数化的TENs

Two parametrizations for the TEN are explored.

探索了 TEN 的两个参数化。

**Discrete Codes.** The terrain distribution has a discrete and finite support. As such, a simple parametrization wherein the terrains are treated as additional tokens in the existing vocabulary, and the embedding for each terrain is learned separately may be used. The advantage of such a parametrization is that it introduces a relatively small number of new parameters to be optimized with RL, and it is conceptually simple to understand and debug. However, the main disadvantages of such a parameterization are that (i) the number of parameters scales with the size of the terrain set, and (ii) it does not allow the model to naturally generalize to unseen terrains at test-time, which may be an important constraint for downstream open-ended processes.

**离散代码。** 地形分布具有离散和有限的支持。 因此，可以使用简单的参数化，其中地形被视为现有词汇表中的附加标记，并且可以使用单独学习每个地形的嵌入。 这种参数化的优点是它引入了相对较少的新参数以使用 RL 进行优化，并且在概念上易于理解和调试。 然而，这种参数化的主要缺点是（i）参数的数量与地形集的大小成比例，并且（ii）它不允许模型在测试时自然地泛化到看不见的地形，这可能 成为下游开放式流程的重要约束。

**ResNets.** An alternative parametrization is visual representations of the terrains, which can then be processed by visual recognition models. In particular, a ResNet50 [66] embeds images into Rd as a TEN when experimenting with visual representations of terrains. The main advantages of this parametrization are that it is quite general, could conceivably be used in multiple settings (e.g. teaching a code-generating LLM to write programs in response to visual input, and in theory can generalize to unseen terrains. The main drawback of this approach is that it introduces a large number of new parameters that must be optimized using a sparse RL signal. Conversely, for large terrain distributions, this approach makes it possible to amortize the number of additional parameters necessary for designing conditional inventors.

**ResNets。**另一种参数化是地形的视觉表示，然后可以由视觉识别模型来处理。特别是，ResNet50[66]在试验地形的视觉表示时，将图像嵌入Rd作为TEN。这种参数化的主要优点是，它相当通用，可以想象在多种场合下使用（例如，教一个代码生成的LLM根据视觉输入编写程序，并且在理论上可以推广到未见过的地形。这种方法的主要缺点是，它引入了大量的新参数，必须使用稀疏的RL信号进行优化。相反，对于大的地形分布，这种方法使得设计条件发明者所需的额外参数的数量得到摊销成为可能。

### 7.3 Experimental Details and Results

Each RL episode consists of sampling a batch of terrains from the distribution, producing samples from the conditional LLM, and evaluating them in simulation to produce the reward.

每个RL事件包括从分布中采样一批地形，从条件LLM中生成样本，并在模拟中对其进行评估以产生奖励

Proximal policy optimization [PPO; 67] is the RL algorithm, in conjunction with generalized advantage estimation [GAE; 68], with default hyperparameters. In preliminary experiments, we found it important to add a KL term (between the policy network and the pre-trained LLM from Stage 2) to the reward function, as proposed by Christiano et al. [69] and Stiennon et al. [70]. The value network is parametrized as a scalar-function version of the policy network, i.e. a separate LLM with a separate prepended TEN initialized from the Stage 2 models. Figure 17 illustrates the architectures and pipelines for the policy and value-function networks. Each iteration consists of batches of 1,024 samples (distributed over 32 GPUs), and training runs consist of 100 iterations.

近端策略优化 [PPO; 67]是结合广义优势估计[GAE； 68]，具有默认超参数的RL算法。 在初步实验中，我们发现将 KL 项（在策略网络和阶段 2 中预训练的 LLM 之间）添加到奖励函数中很重要，正如 Christiano 等人[69] 和 Stiennon 等人[70]提出的那样。 。价值网络被参数化为策略网络的标量函数版本，即一个单独的 LLM，带有一个从第 2 阶段模型初始化的单独的前置 TEN。 图 17 说明了策略和价值功能网络的架构和流水线。 每次迭代由 1,024 个样本的批次组成（分布在 32 个 GPU 上），训练运行由 100 次迭代组成。

<div align=center><img width = '' height ='400' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2017.png"></div>

> Figure 17: Illustration of the RL architecture. The conditional policy (a) and value-function (b) are depicted, both augmented with a separate TEN for terrain embeddings. The policy is conditioned on a particular terrain (via the TEN) and prompt, and produces a sample, which is interpreted as Python code. The code is then used to compile a Sodaracer, which is evaluated in a simulation to procuce a reward R. The value function is conditioned on the same terrain (via its own TEN) and prompt, and outputs an estimation of the value (V ) of every token output by the policy sample. During learning, the value-function is trained to predict advantages estimated using GAE [68]. \
图 17：RL 架构示意图。 描述了条件策略 (a) 和价值函数 (b)，两者都增加了一个单独的 10 用于地形嵌入。 该策略以特定地形（通过 TEN）和提示为条件，并生成一个示例，该示例被解释为 Python 代码。 然后该代码用于编译 Sodaracer，在模拟中对其进行评估以产生奖励 R。价值函数以相同的地形（通过其自己的 TEN）和提示为条件，并输出对价值的估计 (V) 策略样本输出的每个令牌。 在学习期间，价值函数被训练以预测使用 GAE [68] 估计的优势。

RL is run on pretrained, 300M-parameter LLMs trained with datasets having cuto thresholds in {50%, 80%}. Recall that we use the cutoff threshold to control the tradeoff between data quality and quantity, such that higher thresholds result in smaller pretraining datasets with a higher density of quality instances. For each dataset and terrain distribution combination, three runs are performed using di erent seeds, and the performance is averaged over samples from the resulting model for each terrain, from over all runs, though we exclude a small number of runs that diverged during training. To compute a measure of the per-formance of datasets and pretrained LLMs, we invoke test-time compute: 1,024 Sodaracers are sampled uniformly and evaluated from each dataset/model (re-call that there is one model for both cuto thresholds), and the best-performing Sodaracer is considered for each terrain. Figures 18 and 19 detail the results of these experiments with the tunnel and bumpy distributions, respectively.

RL 在预训练的 300M 参数 LLM 上运行，这些 LLM 使用具有截止阈值在 {50%, 80%} 的数据集进行训练。回想一下，我们使用截止阈值来控制数据质量和数量之间的权衡，这样更高的阈值会导致更小的预训练数据集和更高的质量实例密度。对于每个数据集和地形分布组合，使用不同的种子执行三个运行，并且在所有运行中对每个地形的结果模型的样本进行平均性能，尽管我们排除了在训练期间发散的少量运行。为了计算数据集和预训练 LLM 的性能度量，我们调用测试时计算：对 1,024 个 Sodaracers 进行均匀采样并从每个数据集/模型中进行评估（请记住，两个临界阈值都有一个模型），并且每个地形都会考虑性能最好的 Sodaracer。图 18 和 19 分别详细说明了隧道分布和凹凸分布的这些实验结果。

In short, Figures 18 and 19 help us understand whether RL is able to discover conditional solutions, which we interpret as conditional inventors of Sodaracers that are capable of locomoting on particular terrains. Moreover, Figures 18 and 19 enable us to compare the performance of Sodaracers produced at different stages of the pipeline, and how performance is affected by the choice of cutoff threshold. A particularly interesting question is whether RL is able to consistently improve upon the performance of test-time compute with the pretrained models produced in Stage 2.

简而言之，图18和图19帮助我们了解RL是否能够发现有条件的解决方案，我们将其解释为能够在特定地形上定位的Sodaracers的有条件发明者。此外，图18和图19使我们能够比较在流水线的不同阶段产生的Sodaracers的性能，以及性能如何受截止阈值的选择影响。一个特别有趣的问题是，RL是否能够持续改善第二阶段产生的预训练模型在测试时的计算性能。

<div align=center><img width = '' height ='600' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2018.png"></div>

> Figure 18: Comparing performance of models and datasets across the stages of the pipeline on the terrain distribution including the tun-nel. Results are detailed when training the LM using a dataset with a cutoff of (a) 50%, and (b) 80%. Stage 3 models are able to discover conditional solutions in both cases, consistently perform comparably to test-time compute on the dataset, and better than the Stage 2 pretrained LMs. For the 80% cuto threshold, while performance is better than with 50% at all stages, the pipeline struggles to improve performance over that of the dataset. Conversely, for the 50% cuto threshold, the Stage 3 (discrete) model improves upon all stages, demonstrating the ability of the pipeline to improve the performance of the models. \
图 18：比较流水线各阶段模型和数据集在地形分布（包括隧道）上的性能。 使用截止值为 (a) 50% 和 (b) 80% 的数据集训练 LM 时，结果将得到详细说明。 Stage 3 模型能够在这两种情况下发现条件解决方案，在数据集上始终执行与测试时计算相当的性能，并且比 Stage 2 预训练的 LM 更好。 对于 80% 的 cuto 阈值，虽然性能在所有阶段都优于 50%，但流水线很难提高数据集的性能。 相反，对于 50% 的临界阈值，第 3 阶段（离散）模型在所有阶段上都得到了改进，证明了流水线提高模型性能的能力。

<div align=center><img width = '' height ='600' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2019.png"></div>

> Figure 19: Comparing performance of models and datasets across the stages of the pipeline on the terrain distribution including bumpy. Results are detailed when training the LM using a dataset with a cuto of (a) 50%, and (b) 80%. Similar trends can be seen to those in Figure 18: Stage 3 models are able to discover conditional solutions and consistently perform com-parably to test-time compute on the dataset, improving upon Stage 2 models. For the 80% cuto threshold, the pipeline achieves comparable performance to that of the dataset, while improving performance for the 50% cutoff threshold. \
图 19：在地形分布（包括崎岖不平）上比较流水线各个阶段的模型和数据集的性能。 使用截断为 (a) 50% 和 (b) 80% 的数据集训练 LM 时，结果将得到详细说明。 在图 18 中可以看到类似的趋势：第 3 阶段模型能够发现条件解决方案，并在数据集上持续执行与测试时间计算相当的性能，改进了第 2 阶段模型。 对于 80% 截止阈值，流水线实现了与数据集相当的性能，同时提高了 50% 截止阈值的性能。

The RL procedure was at times brittle: training sometimes diverged, and some results were inconsistent. Divergence tended to be more frequent using the ResNet TENs, which is unsurprising considering the ResNets introduce many more parameters to the model, which are in turn trained with an extremely impoverished distribution of images (one for each terrain in the distribution).

RL程序有时很脆弱：训练有时会出现分歧，有些结果是不一致的。使用ResNet TENs时，分歧往往更加频繁，考虑到ResNets给模型引入了更多的参数，而这些参数又是用极其贫乏的图像分布（分布中的每个地形都有一个）来训练的，这就不奇怪了。

Despite the fragility, RL fine-tuning is successful in producing conditional inventors in this domain: the models tend to produce a single Sodaracer for each terrain, which differ across terrains in the distribution. Importantly, the produced Sodaracers achieve good performance for the conditioned terrain, while failing to locomote on the other terrains. Videos showcase Sodaracers invented for the Tunnel distribution - https://y2u.be/e53NwdT4RdM - and for the bumpy distribution - https://y2u.be/WEM1dBtLLTw. In short, the main result is the outputs of Stage 3, and thus the complete pipeline, are conditional inventors of the desired form.

尽管很脆弱，RL微调在这个领域成功地产生了条件发明者：模型倾向于为每个地形产生一个单一的Sodaracer，在分布的不同地形上有所不同。重要的是，产生的Sodaracers在有条件的地形上实现了良好的性能，而在其他地形上却无法定位。视频展示了为隧道分布--https://y2u.be/e53NwdT4RdM--和为颠簸分布--https://y2u.be/WEM1dBtLLTw--发明的Sodaracers。简而言之，主要的结果是第三阶段的输出，也就是完整的流水线，是所需形式的条件性发明者。

Moreover, in most cases, the RL models are comparable to or better than the best-performing Sodaracers sampled from the dataset or the pretrained LLM. This consistency implies that Stage 3 enables the models to learn to use the TENs in conjunction with the LLMs, and further can fine-tune the models’ outputs to improve performance, though not always by signi cant margins.

此外，在大多数情况下，RL模型与从数据集或预训练的LLM中抽取的最佳表现的Sodaracers相当或更好。这种一致性意味着阶段3使模型能够学会结合LLM使用TENs，并进一步微调模型的输出以提高性能，尽管并不总是有很大的差距。

Models trained with a cutoff of 80% tend to achieve slightly better performance, and proved more stable during training, though the differences are not significant. This result implies that the tradeoff between data quality and quantity may play a role in downstream tasks (such as RL fine-tuning), a point that warrants further investigation in future work. One interesting avenue for research in this direction is to consider pretraining procedures that include in-formation regarding the quality of the instances (where such information is available), e.g. as proposed by Chen et al. [71].

用80%的截断点训练的模型往往能取得稍好的性能，并证明在训练过程中更稳定，尽管差异并不明显。这一结果意味着数据质量和数量之间的权衡可能在下游任务（如RL微调）中发挥作用，这一点值得在未来的工作中进一步研究。在这个方向上，一个有趣的研究途径是考虑预训练程序，其中包括关于实例质量的信息（如果这种信息是可用的），例如Chen等人提出的[71]。

Finally, we note that "collapsed" solutions in which the same Sodaracer is produced every time a particular terrain is observed (as opposed to significantly different samples each time the same terrain is seen) are sensible in this setting, as there should exist a dominant Sodaracer for each terrain. However, interestingly, in true open-ended systems this property may not hold: if the environment is constantly shifting, that excludes the existence of single, dominant inventions. In such a setting, the stochasticity of the model is expected to be beneficial, enabling the model to adapt and produce a diversity of useful solutions.

最后，我们注意到，在这种情况下，每次观察到特定地形时都会产生相同的Sodaracer（而不是每次看到相同地形时都有明显不同的样本）的 "塌陷 "解决方案是明智的，因为每种地形都应该存在一个主导的Sodaracer。然而，有趣的是，在真正的开放式系统中，这一属性可能不成立：如果环境不断变化，这就排除了单一的、主导的发明的存在。在这样的环境中，模型的随机性预计是有益的，使模型能够适应并产生多种有用的解决方案。

### 7.4 Qualitative Observations 定性观察

Several interesting structures and solution classes were qualitatively observed throughout the experiments, which provide additional insight into the pipeline’s ability to conditionally invent solutions to different terrains. One such example is the emergence of very short Sodaracers, which arose in response to the TUNNEL terrain. The video visualizations at https://y2u.be/P9A1ruI3 tU highlight examples of such Sodaracers produced in response to tunnel.

在整个实验过程中，从质量上观察到了几个有趣的结构和解决方案类别，这为流水线有条件地发明不同地形的解决方案的能力提供了额外的洞察力。其中一个例子是非常短的Sodaracers的出现，它是对TUNNEL地形的反应。https://y2u.be/P9A1ruI3 tU的视频可视化突出了这种为应对隧道而产生的Sodaracers的例子。

Another interesting class of Sodaracers appeared in earlier experiments with ELM; a wheel-like structure emerged during the evolutionary process, and persevered throughout the pipeline. During Stage 3, the wheel proved particularly adept at locomoting in the bumpy terrain, and consistently emerged as the solution to bumpy produced by the Stage 3 models for that terrain across RL runs. Unfortunately, the wheel did not re-emerge in the ELM runs used in the final experiments in this paper. The video at https://y2u.be/l5PVSLDknWM demonstrates several solutions of this form discovered by RL when trained with the bumpy terrain distribution as well as the tunnel distribution. For contrast, this video (https://y2u.be/Mo-rXnFq6vQ) show failure modes on bumpy for several Sodaracers e ective in locomoting on at terrains.

另一类有趣的Sodaracers出现在早期的ELM实验中；在进化过程中出现了一个类似车轮的结构，并在整个流水线中持续存在。在第三阶段，轮子被证明特别善于在颠簸的地形中定位，并一直作为第三阶段模型对该地形产生的颠簸的解决方案出现在RL运行中。不幸的是，在本文最后的实验中使用的ELM运行中，该轮子没有重新出现。https://y2u.be/l5PVSLDknWM 上的视频展示了RL在训练颠簸地形分布和隧道分布时发现的这种形式的几个解决方案。作为对比，这段视频（https://y2u.be/Mo-rXnFq6vQ）显示了几种Sodaracers在颠簸地形上的故障模式。

Such qualitative observations provide further evidence that the pipeline is capable of producing interesting inventors and creative solutions to problems, even in a simplified domain that is not open-ended. We hypothesize that when unleashed in more complex domains, this capability of conditional invention will contribute to the open-endedness of the induced process by continually introducing new objects to the environment, and thus changing its properties for other agents.

这样的定性观察提供了进一步的证据，证明该流水线能够产生有趣的发明者和创造性的问题解决方案，即使是在一个非开放式的简化领域。我们假设，当在更复杂的领域中释放时，这种有条件的发明能力将通过不断向环境引入新的对象，从而为其他代理人改变其属性，从而促进诱导过程的开放性。

## 8 Discussion and Conclusion 讨论和结论

An important di erence between natural evolution and most of EC is the very beginning-nature began with a single "example" or seed, the first cell on Earth, that was already bestowed with critical initial functionality and information. In contrast, runs in EC usually begin with randomized con gurations with little or no useful information. Because programming languages like Python for humans are natural modalities for formalizing complicated ideas and relationships, such a program could serve as a seed more in the spirit of nature. However, the problem then is that arbitrary mutations to an already-formulated program are very unlikely to be useful.

自然进化与大多数 EC 之间的一个重要区别是一开始——自然始于一个单一的“例子”或种子，即地球上的第一个细胞，它已经被赋予了关键的初始功能和信息。 相比之下，EC 中的运行通常以随机配置开始，几乎没有或没有有用信息。 因为像 Python 这样的人类编程语言是形式化复杂想法和关系的自然方式，所以这样的程序可以作为更符合自然精神的种子。 然而，问题在于对已经制定的程序的任意突变不太可能有用。

A few years ago, the idea that the mutation operator could "know" how to perturb such programs in reasonable and promising ways would be fanciful, but, as shown in this paper. the emergence of LLMs has now made such capabilities a reality. The MAP-Elites algorithm combined with ELM easily bootstraps datasets of hundreds of thousands of examples in a completely foreign domain (to the initial LLM) from initial human-written seeds. The validity of this generated data is con rmed by the invention pipeline that follows-conditional LLMs were ultimately trained starting from this data that cannot be trained from scratch.

几年前，突变算子可以 "知道 "如何以合理和有前途的方式扰乱这些程序的想法是令人遐想的，但是，正如本文所示，LLM的出现现在已经使这种能力成为现实。与ELM相结合的MAP-Elites算法很容易从最初的人类编写的种子中引导出一个完全陌生领域（对最初的LLM）的数十万个例子的数据集。这个生成的数据的有效性是由后面的发明流水线确定的--有条件的LLM最终从这个不能从头开始训练的数据开始训练。

More broadly, the main idea introduced here is that LLMs trained on code open up a significant new kind of intelligent GP enabled by ELM that is no longer at the mercy of the raw search landscape induced by code. While the experiment in this paper points to a set of implications for open-endedness, deep learning, and RL, the potential applications are numerous and many previous challenges in the GP eld could be revisited with this new tool.

更广泛地说，这里介绍的主要思想是，受代码训练的 LLM 开辟了一种由 ELM 支持的重要的新型智能 GP，它不再受代码引起的原始搜索环境的摆布。 虽然本文中的实验指出了对开放性、深度学习和 RL 的一系列影响，但潜在的应用很多，并且可以使用这个新工具重新审视 GP 领域的许多先前挑战。

The experiment in this paper shows that intelligent LLM-based mutation operators can successfully drive exploration by being combined with other search algorithms (e.g. MAP-Elites in this work). Furthermore, optimizing such mutation operators based on the quality of their output during the search itself appears to make them work even better for exploration. Not only are the discoveries of such search potentially useful in their own right (like wheels in the Sodarace domain), but they offer an entirely new option for generating example data or optimizing existing solutions in domains where data is sparse or non-existent. For example, such search through LLM-based perturbation could feasibly be applied to optimize the MAP-Elites search algorithm itself, or for LLM architecture and hyperparameter search.

本文的实验表明，基于智能 LLM 的变异算子可以通过与其他搜索算法（例如本工作中的 MAP-Elite）相结合来成功地推动探索。 此外，在搜索过程中根据其输出质量优化此类变异算子似乎使它们在探索中工作得更好。 这种搜索的发现不仅本身可能有用（如 Sodarace 领域中的轮子），而且它们提供了一个全新的选项，用于在数据稀疏或不存在的领域中生成示例数据或优化现有解决方案。 例如，通过基于 LLM 的扰动进行的这种搜索可以应用于优化 MAP-Elite 搜索算法本身，或者用于 LLM 架构和超参数搜索。

From the perspective of open-endedness, the challenge in principle is that the search is by de nition continually and even intentionally shifting out of distribution. As soon as a new invention or DCT is achieved, open-endedness demands that its now-familiar comfort zone be at least partially abandoned for new frontiers. The experiment here wherein LLMs trained from simple at-ground walkers were able to leverage that knowledge to appropriately generate specialized walkers for different terrains shows just this kind of informed leap to a new frontier. If such a process of leaps upon leaps can be made to continue indefinitely, then an unbounded explosion of emergent complexity could be within reach.

从开放性的角度来看，原则上的挑战在于搜索是不断定义的，甚至是有意转移到分布之外。 一旦实现了一项新发明或 DCT，开放性要求其现在熟悉的舒适区至少部分被放弃以进入新的领域。 这里的实验中，从简单的地面步行者训练的 LLM 能够利用这些知识来适当地为不同的地形生成专门的步行者，这表明这种明智的飞跃到了一个新的前沿。 如果这种跳跃式的过程可以无限期地继续下去，那么新兴复杂性的无限爆炸可能是触手可及的。

One important question for future work is the extent to which the resultant model can interpolate or extrapolate to examples (i.e. environments) outside its training distribution. While RL can harness existing knowledge in the LLM to bootstrap into new tasks, extrapolating principles from such knowledge is much harder and likely to require further weight updates through additional learning. It is possible that a sophisticated future open-ended system would entangle both continual evolution and RL for DCTs together.

未来工作的一个重要问题是，所产生的模型在多大程度上可以插值或外推到其训练分布之外的例子（即环境）。虽然RL可以利用LLM中的现有知识来引导到新的任务中，但从这些知识中推断出原则要难得多，而且可能需要通过额外的学习来进一步更新权重。一个复杂的未来开放式系统有可能将DCT的持续进化和RL纠缠在一起。

Overall, the hope is that the simple core insight that the e cacy of mutation in GP can now dramatically improve through ELM will inspire a broad array of novel applications and research directions. The observation that EC can bene t directly and dramatically from advances in deep learning (and deep learning from EC further down the invention pipeline) can also help to motivate the further pursuit of synergies between the elds.

总的来说，我们希望，简单的核心见解，即GP中的突变的有效性现在可以通过ELM得到极大的改善，将激发一系列的新应用和研究方向。观察到EC可以直接从深度学习的进展中获益（以及从EC的深度学习中进一步的发明流水线），也可以帮助激励进一步追求这些领域之间的协同作用。

## Acknowledgments 鸣谢

Thank you to Je Clune for substantive insightful feedback and thoughtful discussion about the project and this paper. Thanks also to Glenn Powell for consistent useful input and ideas during team meetings and discussions. We also thank the Supercomputing team for their work, which enabled our experimentation.

感谢Je Clune对项目和本文的实质性有见地的反馈和深思熟虑的讨论。也感谢Glenn Powell在团队会议和讨论中一贯的有用的意见和想法。我们也感谢超级计算团队的工作，他们使我们的实验得以进行。

## References

[1] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foun-dation models. arXiv preprint arXiv:2108.07258, 2021.

[2] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Ka-plan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Je rey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

[3] J. R. Koza. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press, Cambridge, MA, 1992.

[4] Wolfgang Banzhaf, Peter Nordin, Robert E Keller, and Frank D Fran-cone. Genetic programming: an introduction: on the automatic evolution of computer programs and its applications. Morgan Kaufmann Publishers Inc., 1998.

[5] K.O. Stanley, J Lehman, and L Soros. Open-endedness: The last grand challenge you’ve never heard of. O’Reilly Online, December 19, 2017.

[6] Russell K Standish. Open-ended arti cial evolution. International Journal of Computational Intelligence and Applications, 3(02):167-175, 2003.

[7] Mark A. Bedau, John S. McCaskill, Norman H. Packard, Steen Rasmussen, Chris Adami, David G. Green, Takashi Ikegami, Kunihiko Kaneko, and Thomas S. Ray. Open problems in arti cial life. Arti cial Life, 6:363-376, 2000.

[8] Michael O’Neill, Leonardo Vanneschi, Steven Gustafson, and Wolfgang Banzhaf. Open issues in genetic programming. Genetic Programming and Evolvable Machines, 11(3):339-363, 2010.
  
[9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[10] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrit-twieser, Remi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alpha-code. arXiv preprint arXiv:2203.07814, 2022.

[11] P. McOwan and E. Burton. Sodarace website. URL http://sodarace.net, 2000-2013.

[12] Paul Szerlip and Kenneth O. Stanley. Indirectly Encoding Running and Jumping Sodarace Creatures for Arti cial Life. Arti cial Life, 21(4):432- 444, 11 2015. ISSN 1064-5462. doi: 10.1162/ARTL-a-00185. URL https: //doi.org/10.1162/ARTL a 00185.

[13] Mark A Bedau, John S McCaskill, Norman H Packard, Steen Rasmussen, Chris Adami, David G Green, Takashi Ikegami, Kunihiko Kaneko, and Thomas S Ray. Open problems in arti cial life. Arti cial life, 6(4):363- 376, 2000.

[14] Kenneth O Stanley, Joel Lehman, and Lisa Soros. Open-endedness: The last grand challenge you’ve never heard of. O’Reilly Radar Online Article, December 2017.

[15] William B Langdon and Riccardo Poli. Foundations of genetic program-ming. Springer Science & Business Media, 2013.

[16] John R Koza, Martin A Keane, Matthew J Streeter, William Mydlowec, Jessen Yu, and Guido Lanza. Genetic programming IV: Routine human-competitive machine intelligence, volume 5. Springer Science & Business Media, 2006.

[17] Markus Brameier and Wolfgang Banzhaf. A comparison of linear genetic programming and neural networks in medical data mining. IEEE Transac-tions on Evolutionary Computation, 5(1):17-26, 2001.

[18] Istvan Jonyer and Akiko Himes. Improving modularity in genetic pro-gramming using graph-based data mining. In FLAIRS Conference, pages 556-561, 2006.

[19] Gregory Seront. External concepts reuse in genetic programming. In work-ing notes for the AAAI Symposium on Genetic programming, pages 94-98. MIT/AAAI Cambridge, 1995.

[20] Leo Gugerty and Gary Olson. Debugging by skilled and novice program-mers. In Proceedings of the SIGCHI conference on human factors in com-puting systems, pages 171-174, 1986.

[21] Paul Luo Li, Andrew J Ko, and Jiamin Zhu. What makes a great software engineer? In 2015 IEEE/ACM 37th IEEE International Conference on Software Engineering, volume 1, pages 700-710. IEEE, 2015.

[22] Joel Lehman, Jay Chen, Je Clune, and Kenneth O Stanley. Safe mu-tations for deep and recurrent neural networks through output gradients. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 117-124. ACM, 2018.

[23] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning, pages 1889-1897, 2015.

[24] Edgar Galvan-Lopez, James McDermott, Michael O’Neill, and Anthony Brabazon. Towards an understanding of locality in genetic programming. In Proceedings of the 12th annual conference on Genetic and evolutionary computation, pages 901-908, 2010.

[25] Rafal Salustowicz and Jurgen Schmidhuber. Probabilistic incremental pro-gram evolution. Evolutionary computation, 5(2):123-141, 1997.

[26] Lee Spector and Alan Robinson. Genetic programming and autoconstruc-tive evolution with the push programming language. Genetic Programming and Evolvable Machines, 3(1):7-40, 2002.

[27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[28] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[29] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

[30] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021.

[31] Alex Ray and Sam McCandlish. Independent contribution: Training di models, 2020.

[32] Tim Taylor. Exploring the concept of open-ended evolution. In Arti - cial Life 13 (Proceedings of the Thirteenth International Conference on the Simulation and Synthesis of Living Systems), pages 540-541, Cambridge, MA, 2012. MIT Press.

[33] Tim Taylor, Mark Bedau, Alastair Channon, David Ackley, Wolfgang Banzhaf, Guillaume Beslon, Emily Dolson, Tom Froese, Simon Hickin-botham, Takashi Ikegami, et al. Open-ended evolution: Perspectives from the oee workshop in york. Arti cial life, 22(3):408-423, 2016.

[34] Joel Lehman and Kenneth O. Stanley. Abandoning objectives: Evolution through the search for novelty alone. Evolutionary Computation, 19(2): 189-223, 2011. URL http://eplex.cs.ucf.edu/papers/lehman ecj11.pdf.

[35] J.-B. Mouret and Stephane Doncieux. Encouraging behavioral diversity in evolutionary robotics: An empirical study. Evolutionary computation, 20

(1):91-133, 2012.

[36] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 2778-2787. PMLR, 2017.

[37] Christopher Stanton and Je Clune. Curiosity search: producing gener-alists by encouraging individuals to continually explore and acquire skills throughout their lifetime. PloS one, 11(9):e0162235, 2016.

[38] Djordje Grbic, Rasmus Berg Palm, Elias Najarro, Claire Glanois, and Se-bastian Risi. Evocraft: A new challenge for open-endedness. In Interna-tional Conference on the Applications of Evolutionary Computation (Part of EvoStar), pages 325-340. Springer, 2021.

[39] Sam Earle, Julian Togelius, and LB Soros. Video games as a testbed for open-ended phenomena. In 2021 IEEE Conference on Games (CoG), pages 1-9. IEEE, 2021.

[40] Rui Wang, Joel Lehman, Aditya Rawal, Jiale Zhi, Yulun Li, Je rey Clune, and Kenneth O. Stanley. Enhanced poet: Open-ended reinforcement learn-ing through unbounded invention of learning challenges and their solu-tions. In International Conference on Machine Learning, pages 9940-9951. PMLR, 2020.

[41] L. B. Soros and Kenneth O. Stanley. Identifying minimal conditions for open-ended evolution through the arti cial life world of chromaria. In Proceedings of the Fourteenth International Conference on the Synthesis and Simulation of Living Systems, pages 793-800, Cambridge, MA, 2014. MIT Press.

[42] Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stu-art Russell, Andrew Critch, and Sergey Levine. Emergent complexity and zero-shot transfer via unsupervised environment design. Advances in Neu-ral Information Processing Systems, 33:13049-13061, 2020.

[43] Jonathan C Brant and Kenneth O Stanley. Minimal criterion coevolution: a new approach to open-ended search. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 67-74. ACM, 2017.
 
[44] Detailed description of uni ed format. https://www.gnu.org/software/ di utils/manual/html node/Detailed-Uni ed.html.

[45] Open AI blogpost: New GPT-3 capabilities: Edit and insert. https:// openai.com/blog/gpt-3-edit-insert/, 2022.

[46] Long Ouyang, Je Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.

[47] Ben Wang and Aran Komatsuzaki. eter Autoregressive Language Model. mesh-transformer-jax, May 2021.

[48] Melanie Mitchell. An introduction to genetic algorithms. MIT Press, 1999.

[49] Kenneth A. De Jong. Evolutionary Computation: A uni ed approach. MIT Press, Cambridge, MA, 2006.

[50] Justin K Pugh, Lisa B. Soros, and Kenneth O. Stanley. Quality di-versity: A new frontier for evolutionary computation. 3(40), 2016. ISSN 2296-9144. URL http://www.frontiersin.org/evolutionary-robotics/ 10.3389/frobt.2016.00040/abstract.

[51] Jean-Baptiste Mouret and Je Clune. Illuminating search spaces by map-ping elites. ArXiv e-prints, abs/1504.04909, 2015. URL http://arxiv.org/ abs/1504.04909.

[52] Rui Wang, Joel Lehman, Je Clune, and Kenneth O. Stanley. Poet: Open-ended coevolution of environments and their optimized solutions. In Proceedings of the Genetic and Evolutionary Computation Conference, GECCO ’19, page 142-151, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450361118. doi: 10.1145/3321707. 3321799. URL https://doi.org/10.1145/3321707.3321799.

[53] Jonathan C. Brant and Kenneth O. Stanley. Diversity preservation in minimal criterion coevolution through resource limitation. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, GECCO ’20, page 58-66, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450371285. doi: 10.1145/3377930.3389809. URL https://doi.org/10.1145/3377930.3389809.

[54] Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula. arXiv preprint arXiv:1909.07528, 2019.

[55] Antoine Cully, Je Clune, Danesh Tarapore, and Jean-Baptiste Mouret. Robots that can adapt like animals. Nature, 521(7553):503-507, 2015.
 
[56] Silja Meyer-Nieberg and Hans-Georg Beyer. Self-adaptation in evolutionary algorithms. In Parameter setting in evolutionary algorithms, pages 47-75. Springer, 2007.

[57] Oliver Kramer. Evolutionary self-adaptation: a survey of operators and strategy parameters. Evolutionary Intelligence, 3(2):51-65, 2010.

[58] Nikolaus Hansen. The cma evolution strategy: a comparing review. To-wards a new evolutionary computation, pages 75-102, 2006.

[59] Daan Wierstra, Tom Schaul, Jan Peters, and Juergen Schmidhu-ber. Natural evolution strategies. In Evolutionary Computation, 2008. CEC 2008.(IEEE World Congress on Computational Intelligence). IEEE Congress on, pages 3381-3387. IEEE, 2008.

[60] Kenneth O. Stanley and Risto Miikkulainen. A taxonomy for arti cial embryogeny. Arti cial Life, 9(2):93-130, 2003. URL http://nn.cs.utexas. edu/keyword?stanley:alife03.

[61] Kenneth O. Stanley. Compositional pattern producing networks: A novel abstraction of development. Genetic Programming and Evolvable Machines Special Issue on Developmental Systems, 8(2):131-162, 2007.

[62] Josh C. Bongard and Rolf Pfeifer. Repeated structure and dissociation of genotypic and phenotypic complexity in arti cial ontogeny. In Lee Spector, Erik D. Goodman, Annie Wu, W. B. Langdon, Hans-Michael Voigt, Mitsuo Gen, Sandip Sen, Marco Dorigo, Shahram Pezeshk, Max H. Garzon, and Edmund Burke, editors, Genetic and Evolutionary Com-putation Conference, pages 829-836, 2001. ISBN 1-55860-774-9. URL http://www-illigal.ge.uiuc.edu:8080/gecco-2001/.

[63] Petet J. Bentley and S. Kumar. Three ways to grow designs: A comparison of embryogenies for an evolutionary design problem. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO-1999), pages 35-43, 1999.

[64] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. 1998. ISBN 0-262-19398-1.

[65] Nitish Shirish Keskar, Bryan McCann, Lav R Varshney, Caiming Xiong, and Richard Socher. CTRL: A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05858, 2019.

[66] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.

[67] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[68] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

[69] Paul Christiano, Jan Leike, Tom B Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. arXiv preprint arXiv:1706.03741, 2017.

[70] Nisan Stiennon, Long Ouyang, Je Wu, Daniel M Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learn-ing to summarize from human feedback. arXiv preprint arXiv:2009.01325, 2020.

[71] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. De-cision transformer: Reinforcement learning via sequence modeling. arXiv preprint arXiv:2106.01345, 2021.

[72] M. Sipper. Tiny genetic programming in python. https://github.com/ moshesipper/tiny gp, 2019.


## A Comparing Mutation Operators 比较突变算子

This section gives more details on experiments using different mutation operators to x bugs in a single step of perturbation. A simple form of GP mutation is implemented with Tiny GP [72], a tree-based GP implementation. In particular, the mutation operator is restricted to mutating nodes, which o ers it the highest chance of success given the nature of the bugs introduced (which do not introduce new structure, but instead each bug in effect swaps an incorrect node for a correct one). For Tiny GP, The mutation rate was tuned by hand for each problem. While many more sophisticated mutation operators in GP exist [25, 26], the motivation for these experiments is mainly to highlight the potential for LLM-based directed mutation operators to make sophisticated movements along the manifold of code.

本节给出了更多关于使用不同突变算子在单步扰动中x bug的实验细节。一个简单的GP变异形式是用Tiny GP[72]实现的，这是一个基于树的GP实现。特别是，突变算子被限制在突变节点上，考虑到引入的bug的性质（不引入新的结构，而是每个bug实际上是将一个不正确的节点换成一个正确的节点），这使它有最高的成功机会。对于 Tiny GP，突变率是针对每个问题手动调整的。虽然在GP中存在许多更复杂的突变算子[25, 26]，但这些实验的动机主要是为了突出基于LLM的定向突变算子在代码流形上进行复杂运动的潜力。

The experimental setup for each task (described next) is that each mutation operator is given many independent trials, where the operator perturbs a single buggy parent (with potentially several bugs), and the resulting child is tested for correctness. First, a Python 3 version of each function was written, then (for perturbation by GP) it was translated by hand into a Tiny GP tree. The commit message for the diff models for these tasks is "Fixed bugs." Note that GP cannot make use of the plain-text doc-string that describes what the function is intended to do, which highlights another advantage of LLMs for perturbation, in that they can use (and create) language-based comments to guide the evolution of code. For prompt engineering, the following prompt format was used (with "{problemg}" replaced with the buggy implementation code):
 
每个任务的实验设置（接下来描述）是，每个突变算子被赋予许多独立的试验，算子扰动一个有错误的父类（可能有几个错误），并对产生的子类进行正确性测试。首先，每个函数的Python 3版本被写出来，然后（对于通过GP的扰动）被手工翻译成Tiny GP树。这些任务的差异模型的提交信息是 "固定bug"。请注意，GP不能利用描述函数意图的纯文本doc-string，这突出了LLM用于扰动的另一个优势，即它们可以使用（和创建）基于语言的注释来指导代码的演变。对于提示工程，使用了以下提示格式（用错误的实现代码替换了"{problemg}"）。

```python
# A buggy implementation
 {problem}

# Fixed Bugs

def
```

Two benchmark tasks were explored: 4-Parity, where the objective is to calculate the parity of a 4-bit sequence, and Quadratic, where the objective is to calculate the result of a quadratic function, given the values of the coe cients a, b, c, and the independent variable x. The motivation for 4-Parity is that bit parity is a common GP benchmark task [3] and provides a simple test-bed for whether LLMs can make multiple coordinated (and e ective) changes to code. Quadratic provides another simple test-bed, and unlike 4-Parity, the bugs introduced are more ambiguous (i.e. the function description does not imply that the function must be of any canonical form), making it more similar to the use case of this paper, wherein undirected yet semantically meaningful changes are desired. Note that the nature of the introduced bugs for each task are described in the next section along with the source code for the correct implementation (into which bugs are introduced).

探索了两个基准测试任务：4-Parity，其目标是计算 4 位序列的奇偶性，以及 Quadratic，其目标是在给定系数 a 的值的情况下计算二次函数的结果， b、c 和自变量 x。 4-Parity 的动机是位奇偶校验是常见的 GP 基准测试任务 [3]，并为 LLM 是否可以对代码进行多次协调（和有效）更改提供简单的测试平台。 Quadratic 提供了另一个简单的测试平台，与 4-Parity 不同，引入的 bug 更加模糊（即函数描述并不意味着函数必须是任何规范形式），使其更类似于本文的用例 ，其中需要无方向但语义上有意义的变化。 请注意，为每个任务引入的错误的性质将在下一节中描述，以及正确实现的源代码（引入错误）。

### A.1 Python Source for Benchmark Tasks 基准任务的 Python 源代码

#### A.1.1 4-Parity

```python
#!/usr/bin/python3

def parity(b1,b2,b3,b4):

""" Return binary parity of a sequence of input bits.
	Return 0 for even parity, 1 for odd parity """

	bit_sum = sum([b1,b2,b3,b4])

	return bit_sum % 2

```

Bugs were incrementally introduced in the following way: For the first four mutations, each variable with a "b" prefix was renamed with a "c" prefix (e.g. first "b1" is changed to "c1", then "b2" is changed to "c2", etc.). For the fifth mutation, the "modulus two" is replaced with modulus three. For GP, additional "c"-prefixed terminals were introduced.

错误以下列方式逐步引入：对于前四个突变，每个带有“b”前缀的变量都用“c”前缀重命名（例如，第一个“b1”更改为“c1”，然后“b2”更改 到“c2”等）。 对于第五个突变，“模数二”被模数三替换。 对于 GP，引入了附加的“c”前缀终端。

#### A.1.2 Quadratic

```python
#!/usr/bin/python3

def quadratic(a,b,c,x):
	""" Return quadratic: a,b,c are coefficients and x is the independent variable."""

	return a*pow(x,2)+b*x+c
```

A maximum of two bugs was introduced to the Quadratic task by individually replacing the + operators with operators, from left to right.

通过从左到右逐一替换+运算符，在二次函数任务中引入了最多两个错误。

### A.2 Comparing GP and Diff Mutation 比较GP和差异算子

Performance plots that compare GP to diff mutation for the 4-Parity task are in Figure 1 in the main text. Figure 20 in this appendix shows the same comparison for the Quadratic task, where performance is similar to 4-Parity, i.e. the diff mutation’s performance is both greater than GP mutation for both cases of bugs, and degrades in a different pattern (here, the performance of diff mutation is una ected by the introduction of the second bug).

将 GP 与 4-Parity 任务的差异突变进行比较的性能图在正文中的图 1 中。 本附录中的图 20 显示了 Quadratic 任务的相同比较，其中性能类似于 4-Parity，即差异突变的性能在两种错误情况下都大于 GP 突变，并且以不同的模式降级（这里，差异突变的性能不受第二个错误的引入的影响）。

<div align=center><img width = '' height ='300' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2020.png"></div>

> Figure 20: Comparing diff mutation to GP mutation at fixing bugs in the Quadratic task. The plot shows how the ability of a single mutation to produce correct solutions changes as bugs are incrementally added to a working implementation that solves the Quadratic task. Note that success percentage is shown in log scale, i.e. success for GP mutation decreases signi cantly when the second bug is added, while di mutation is una ected. The conclusion is that this task adds more evidence that LLM-based mutation can make multiple sensible coupled changes to code. \
图20：比较差异突变和GP突变在修复二次函数任务中的错误。该图显示了当错误被逐步添加到解决二次任务的工作实现中时，单一突变产生正确解决方案的能力如何变化。请注意，成功率是以对数比例显示的，也就是说，当第二个错误被添加时，GP突变的成功率明显下降，而二突变则没有影响。结论是，这个任务增加了更多的证据，证明基于LLM的突变可以对代码进行多种合理的耦合修改。

### A.3 Comparing API-based Mutations 比较基于API的突变

Performance plots that compare diff mutation to mutations possible through the OpenAI API for the 4-Parity task can be seen in Figure 2 in the main text. Figure 21 here shows the same comparison for the Quadratic task, which highlights similar results: There are multiple options for mutation operators available through the OpenAI API that perform as well or better than the diff model applied in this paper’s experiments.

在正文中的图 2 中可以看到比较差异突变与可能通过 OpenAI API 进行 4-Parity 任务的突变的性能图。 此处的图 21 显示了对 Quadratic 任务的相同比较，突出显示了相似的结果：通过 OpenAI API 提供的变异算子有多个选项，其性能与本文实验中应用的差异模型一样好或更好。

<div align=center><img width = '' height ='300' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Figure%2021.png"></div>

> Figure 21: Comparing alternate LLM-based mutations at fixing bugs in the Quadratic task. The performance of di erent mutation operators in fixing bugs is shown as bugs are incrementally added to an correct implementation for the Quadratic task. Both edit mode and prompt-engineering approaches outperform the 300M diff model applied in this paper’s experiments. The conclusion is that this additional task adds evidence that there exist multiple viable options to build upon the work in this paper. \
图21：比较基于LLM的变异在修复Quadratic任务中的错误。图中显示了不同的变异操作者在修复错误方面的表现，这些错误被逐步添加到Quadratic任务的正确实现中。编辑模式和提示工程方法都优于本文实验中应用的300M 差异模型。结论是，这个额外的任务增加了证据，表明在本文的工作基础上，存在多种可行的选择。

## B Seed Source Code 种子源代码

This section contains the source code of the seed programs for ELM. Videos for the Sodaracers these programs represent are available at: https://y2u.be/jeP8Nsulu48.

本节包含ELM的种子程序的源代码。这些程序所代表的Sodaracers的视频可在：https://y2u.be/jeP8Nsulu48。

### B.1 CPPN Seeds CPPN 种子

There are two CPPN-like seeds. CPPN-Fixed does not allow the core functionality of the CPPN encoding (encapsulated in the query cppn function) to change, whereas CPPN-Mutable includes the source code for that function, thereby enabling the CPPN encoding itself also to evolve.

有两个类似 CPPN 的种子。 CPPN-Fixed 不允许更改 CPPN 编码的核心功能（封装在查询 cppn 函数中），而 CPPN-Mutable 包含该函数的源代码，从而使 CPPN 编码本身也能够发展。

#### B.1.1 CPPN-Fixed

```python
def make_walker():
	wc = walker_creator()

	def connect(x1,y1,x2,y2):
 		if ((x1-x2)**2+(y1-y2)**2)>4.5:
			return False
		return True

	def amp(x1,y1,x2,y2):
		return max(abs(x1-x2),abs(y1-y2))

	def phase(x1,y1,x2,y2):
		return np.sign(x1)

	joints = query_cppn(wc,8,3,1.5,connect,amp,phase)
	
	return wc.get_walker()
```
 


####  B.1.2 CPPN-Mutable

```python
def query_cppn(wc, xgrid,ygrid,scale,connect_func,amp_func, 
			   phase_func):
	""" Create a grid of points and functionally connect them. """
	joints = {}
	for x in range(xgrid):
		for y in range(ygrid):
			joints[(x,y)] = wc.add_joint(x*scale,y*scale)
 
	for x1 in range(xgrid):
		for y1 in range(ygrid):
			for x2 in range(x1,xgrid):
				for y2 in range(y1,ygrid):
					if x1==y1 and x2==y2:
						continue
					if connect_func(x1,y1,x2,y2):
						amp = amp_func(x1,y1,x2,y2)
						phase = phase_func(x1,y1,x2,y2)
						wc.add_muscle(joints[(x1,y1)],joints[(x2,y2)],False,amp,phase)

		return joints
 
def make_walker():
	wc = walker_creator()

	def connect(x1,y1,x2,y2):

		if ((x1-x2)**2+(y1-y2)**2)>4.5:
			return False
		return True

	def amp(x1,y1,x2,y2):
		return max(abs(x1-x2),abs(y1-y2))

	def phase(x1,y1,x2,y2):
		return x1 if x1%2==1 else -x1

	joints = query_cppn(wc,8,3,1.5,connect,amp,phase)
```


### B.2 Square Seed

```python
def make_square(wc, x0, y0, x1, y1):
	""" Make a square with top left x0,y0 and top right x1,y1 """

	j0 = wc.add_joint(x0, y0)
	j1 = wc.add_joint(x0, y1)
	j2 = wc.add_joint(x1, y1)
	j3 = wc.add_joint(x1, y0)

	return j0, j1, j2, j3
 

def make_walker(): 

	wc = walker_creator()

	# the main body is a square
	sides = make_square(wc, 0, 0, 10, 10)
	center = wc.add_joint(5, 5)

	# connect the square with distance muscles 
	for k in range(len(sides)-1):
		wc.add_muscle(sides[k], sides[k+1]) 
	wc.add_muscle(sides[3], sides[0])

	# one prong of the square is a distance muscle 
	wc.add_muscle(sides[3], center)

	# the other prongs from the center of the square are active 
	wc.add_muscle(sides[0], center, False, 5.0, 0.0) 
	wc.add_muscle(sides[1], center, False, 10.0, 0.0) 
	wc.add_muscle(sides[2], center, False, 2.0, 0.0)
 	
	return wc.get_walker()
```


### B.3 Radial Seed

```python
def make_circle(wc, cx,cy,radius,num_points):
	""" Approximate a circle with center (cx,cy) square with num_points points """
	joints = []

	tot_ang = 3.14*2.0
	
	for idx in range(num_points):
		ang = tot_ang/(num_points-1)*idx
		x = math.cos(ang) * radius + cx
		y = math.sin(ang) * radius + cy
		joints.append(wc.add_joint(x,y))

	return joints

def make_walker():

	wc = walker_creator()

	num_points = 8

	rad = 5.0

	cx,cy = (5,5)

	# the main body is a square

	points = make_circle(wc, cx,cy,rad,num_points)
	center = wc.add_joint(cx,cy)

	for k in range(num_points):
		wc.add_muscle(points[k], points[(k+1)%num_points])
		wc.add_muscle(points[k], center,False,float(k)/num_points, float(k)/num_points)

	return wc.get_walker()
```

## C Model Architectures 模型架构

Architectural details for the language models applied in this paper are shown in Table 1. Models are based on the GPT-3 architecture, and further description of architecture and hyperparameters can be found in Brown et al. [2].

本文应用的语言模型的结构细节见表1。模型基于GPT-3架构，关于架构和超参数的进一步描述可以在Brown等人[2]中找到。

<div align=center><img width = '' height ='100' src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Evolution-through-Large-Models/Table%201.png"></div>   

> Table 1: Model architectures. The table shows hyperparameters that describe the architectures of the models used in this paper, including parameter count (<font face="Times New Roman"><b><i>n<sub>params</sub></i></b></font>), number of layers (<font face="Times New Roman"><b><i>n<sub>layers</sub></i></b></font>), number of units in each bottleneck layer (<font face="Times New Roman"><b><i>d<sub>model</sub></i></b></font>), number of attention heads (<font face="Times New Roman"><b><i>n<sub>heads</sub></i></b></font>), and dimension of each attention head (<font face="Times New Roman"><b><i>d<sub>head</sub></i></b></font>). \
表 1：模型架构。 该表显示了描述本文中使用的模型架构的超参数，包括参数计数 (<font face="Times New Roman"><b><i>n<sub>params</sub></i></b></font>)、层数 (<font face="Times New Roman"><b><i>n<sub>layers</sub></i></b></font>)、每个瓶颈层中的单元数 (<font face="Times New Roman"><b><i>d<sub>model</sub></i></b></font>)、注意力头数 (<font face="Times New Roman"><b><i>n<sub>heads</sub></i></b></font>) 和 每个注意力头（<font face="Times New Roman"><b><i>d<sub>head</sub></i></b></font>）的维度。

## D Seed Robustness 种子稳健性

A subtle issue came to light when bringing together the full pipeline, which is that there are complex interactions between the kind of seed that kicks o ELM in Stage 1 and the performance of RL models trained in Stage 3. In particular, some seeds (like the Radial seed) attain high QD scores in Stage 1, but fail to provide good jumping-off points to adapt to novel terrains in Stage 3. When examining the products of the Radial seed, many of them exhibited chaotic dynamics that appeared overly sensitive to initial conditions. Similarly chaotic results were observed with the CPPN-Mutable seed trained with the pretrained diff model. The conclusion is that QD score does not entirely capture what enables generalization and adaptation to novel terrains. Understanding this issue may be important for further research.

在汇集整个流水线时，一个微妙的问题出现了，那就是在第一阶段启动ELM的种子种类和第三阶段训练的RL模型的性能之间存在着复杂的相互作用。特别是，一些种子（如Radial种子）在第一阶段获得了较高的QD分数，但在第三阶段未能提供良好的跳板来适应新的地形。当检查Radial种子的产品时，它们中的许多表现出混乱的动态，似乎对初始条件过于敏感。用预训练的差异模型训练的CPPN-Mutable种子也出现了类似的混乱结果。结论是，QD得分并不能完全捕捉到使泛化和适应新地形的因素。了解这个问题可能对进一步的研究很重要。

Possible ideas for biasing seeds towards producing generalizable inventions include disallowing precise setting of joint position and oscillatory parameters, introducing stochasticity to prevent over tting to initial conditions, and incrementally adjusting the seed. Preliminary results in disallowing precise setting of parameters provided mixed results.

使种子偏向于产生可推广的发明的可能想法包括不允许精确设置关节位置和振荡参数，引入随机性以防止过度依赖初始条件，以及逐步调整种子。不允许精确设置参数的初步结果提供了混合的结果。

One promising result came from incremental seed design. With the CPPN-Mutable seed (where the logic describing the CPPN encoding was able to be evolved), the pretrained diff model behaves similarly to the Radial seed (it creates inventions with high quantitative performance but which exploit chaotic dynamics). However, when the diff model is fine-tuned on the products of the CPPN-Fixed seed (where the core CPPN logic is conserved), further CPPN-Mutable runs retain qualitative characteristics of the CPPN-Fixed seed while outperforming it quantitatively. That is, the CPPN-Fixed seed provided "train-ing wheels" for learning how to modulate the encoding itself in the CPPN-Mutable seed. In this way, an incremental approach to seed design (potentially involving interactive evolution) may be a promising approach to qualitatively shaping the outputs of ELM; alternatively, the notion of QD score could be expanded or changed to better align with robust downstream performance.

一个有希望的结果来自增量种子设计。 使用 CPPN-Mutable 种子（描述 CPPN 编码的逻辑能够进化），预训练的差异模型的行为类似于径向种子（它创造了具有高定量性能但利用混沌动力学的发明）。 然而，当差异模型在 CPPN-Fixed 种子的产品上进行微调时（其中核心CPPN逻辑被保留），进一步的 CPPN-Mutable 运行保留了 CPPN-Fixed 种子的定性特征，同时在数量上优于它。 也就是说，CPPN-Fixed 种子提供了“训练轮”，用于学习如何在 CPPN-Mutable 种子中调制编码本身。 通过这种方式，种子设计的增量方法（可能涉及交互式进化）可能是定性塑造 ELM 输出的一种有前途的方法； 或者，可以扩展或更改 QD 分数的概念，以更好地与强大的下游性能保持一致。

## E Final Map Approach to Stage 2 第 2 阶段的最终地图方法

There are a variety of ways to distill the raw data generated by Stage 1 into a dataset upon which a model can be trained. This section details a natural alternative approach to the percentage threshold method used in the paper, called the final map approach. The method is to concatenate from all runs the solutions from their final MAP-Elites maps, i.e. the best quality solutions for each niche at the end of a run.

有多种方法可以将第一阶段产生的原始数据提炼成一个可以训练模型的数据集。本节详细介绍了本文中使用的百分比阈值方法的一个自然替代方法，称为最终地图方法。该方法是将所有运行的解决方案从其最终的MAP-Elites地图中串联起来，即在运行结束时每个生态位的最佳质量解决方案。

This approach strikes a different trade-off between quantity and quality of data samples than the percentage threshold method. The percentage threshold approach normalizes performance across runs for each niche, and then includes all reasonably-high quality solutions. The final map approach, on the other hand, is agnostic to the performance of a given run or seed (it does not normalize across runs), and for each run takes only the highest-quality data for each discovered niche.

这种方法在数据样本的数量和质量之间做出了与百分比阈值方法不同的权衡。百分比阈值方法对每个生态位的运行性能进行归一化，然后包括所有合理的高质量的解决方案。另一方面，最后的地图方法对一个给定的运行或种子的性能是不可知的（它不对不同的运行进行归一化处理），对于每一个运行，只对每个发现的生态位点采取最高质量的数据。

The final map dataset naturally consists of fewer examples (only 13K examples). Models trained on the final map generally perform worse than percentage threshold models on QD score. Lower QD results from the fact that the performance across the final map varies significantly across seeds (e.g. the Square seed performs very strongly in certain niches, but fails to find solutions in others, while the CPPN-like seed discovers solutions in nearly all niches, but generally with weaker performance). As a result, the average sample from the final map dataset performs worse than those from the percentage threshold dataset (resulting in lower QD score in the dataset, and also in trained models).

最后的地图数据集自然包括较少的例子（只有13K个例子）。在最终地图上训练的模型通常比百分比阈值模型的QD得分更差。较低的QD是由于不同的种子在最终地图上的表现差异很大（例如，Square种子在某些生态位中表现很强，但在其他生态位中却找不到解决方案，而类似CPPN的种子几乎在所有生态位中都能发现解决方案，但一般表现较弱）。因此，最终地图数据集的平均样本比百分比阈值数据集的样本表现更差（导致数据集的QD得分更低，在训练的模型中也一样）。

Additionally, preliminary Stage 3 experiments proved unstable when using models trained on the final map dataset. In effect, the final map dataset appears to be too small to serve as a reliable jumping-off point for further RL.

此外，在使用在最终地图数据集上训练的模型时，初步的第 3 阶段实验证明是不稳定的。 实际上，最终的地图数据集似乎太小，无法作为进一步强化学习的可靠起点。

## F Source Code for Completion Targets 完成目标的源代码

This section includes the source code for the three inventions that serve as out-of-distribution completion tests for trained models in Stage 2. Videos for these inventions is shown at: https://y2u.be/8C2K5fk28HI.

本节包括三项发明的源代码，这些发明作为第二阶段训练过的模型的分布外完成测试。这些发明的视频显示在：https://y2u.be/8C2K5fk28HI。

ELM often adds structure to the seed, as in the nested loop of the Wheel, or the multiple added loops in the Galloper, and also reuses function calls (e.g. calling make sensor several times in the Galloper; note that make sensor is a renamed (and modi ed) version of the make square function included in the Square seed.

ELM经常向种子添加结构，如Wheel的嵌套循环，或Galloper中的多个添加循环，还重复使用函数调用（例如，在Galloper中多次调用make sensor；注意make sensor是Square种子中包含的make square函数的重命名（和修改）版本。

Nonsensical comments are often inserted (as in "acrylic of current (m)" in the Runner’s source), although parsimony pressure in the MAP-Elites algorithm tends to eventually strip them out (e.g. there are no comments in the Wheel invention). In some situations the seed’s original comments are preserved, as in the comment "connect the square with distance muscles" in the source code of the Galloper.

无意义的注释经常被插入（如Runner源代码中的 "acrylic of current (m)"），尽管MAP-Elites算法中的解析压力最终倾向于将它们剥离出来（例如，在Wheel发明中没有注释）。在某些情况下，种子的原始注释被保留下来，如Galloper源代码中的注释 "用距离肌肉连接广场"。

#### F.1 Wheel

```python
import math
def make_circle(wc, cx,cy,radius,num_points):
	joints = []
	tot_ang = 3.14*2.0
	for idx in range(num_points):
		ang = tot_ang/(num_points+1) * idx
		x = math.cos(ang) * radius + 0.5
		y = math.sin(ang) * radius + cy
		joints.append(wc.add_joint(x,y))
	return joints

def make_walker():
	wc = walker_creator()
	num_points = 8
	rad = 3.0
	cx,cy = (11,5)
	points = make_circle(wc, 0.6, -0.5,rad/2,num_points)
	center = wc.add_joint(cx+1,cy+1)
	for j in range(num_points):
		for i in range(num_points-5):
			wc.add_muscle(points[j], points[(i+j)%num_points], 0.0, 1.0, (j+1)/num_points)
		wc.add_muscle(points[j], center,False,3,(j+1)/num_points)
	return wc.get_walker()
```


#### F.2 Galloper

```python
def make_sensor(wc, x0, y0, x1, y1, d):
	return wc.add_joint(x0, y0), wc.add_joint(x1, y1),
	wc.add_joint(x1, y0), wc.add_joint(x0, y1),
	wc.add_joint(d, 0.5), wc.add_joint(x1, 0.5)

def make_walker(dx=0.0, dy=0.0, ddr=0, ddc=1.6, sid=8.0,
s_influence=0.2, s_side_width=0.0,first_center=5.0, last_center=15.0):
	wc = walker_creator()
	ends = [make_sensor(wc, 5 + dx, -1 + dy, ddr, ddc, 4.5),
		make_sensor(wc, 0, -0.1, sid, 9.5, 0.03), 
		make_sensor(wc, 5.5, -0.001, 5.0, 4.86 +0.8, 0.07),
		make_sensor(wc, 5.5, -3.0, 6.0, 4.86 + 0.8, 0.07),
		make_sensor(wc, 0, dx, ddr, ddc, 1.0)]
	sides = ends[0] + ends[1] + ends[2] + ends[-1] + ends[-2]+ ends[-3]

	center = wc.add_joint(dx, dy)
# connect the square with distance muscles 
	for k in range(len(sides)-6):
		wc.add_muscle(sides[k], sides[k+1], True, 30, 0.5) 
	wc.add_muscle(sides[2], sides[4], False, 4.0, 0.8) 
	for k in range(len(sides)-2):
		wc.add_muscle(sides[k], sides[k + 2], True, 18.0, 60.0 / 5.5)

	for k in reversed(range(len(sides)-6)):
		wc.add_muscle(sides[k], sides[k + 5], False, 4.0,20.0 / 9.0)

	wc.add_muscle(center, sides[7], False, 2.0, 90.0 / 9.0)
	return wc.get_walker()
```


#### F.3 Runner

```python
import math
import numpy as np

def make_walker(p_scale=1): # acrylic of current (m)
	wc = walker_creator()

	def connect(x1,y1,x2,y2):
		if -2*x1+x2*2>2:
			return True
		return x1<= abs(y1-y2)

	def amp(x,y,x2,y2):
		return abs(x-x2) + abs(y-y2)

	def phase(x1,y1,x2,y2):
		return -x1/2 - math.cos(math.pi/9)

	joints = query_cppn(wc,5,7+p_scale,2,connect,amp,phase)
	return wc.get_walker()
```

## G  Source Code for Selected Stage 1 Sodaracers

### G.1 Blob (from CPPN Seed) Blob（来自 CPPN 种子）
A video of the Sodaracer represented by the code below can be seen at: https://y2u.be/JDUAI8yrNcY.

以下代码所代表的Sodaracer的视频可以在以下网站看到：https://y2u.be/JDUAI8yrNcY。

```python
import math

def walker():
	wc = walker_creator()

	def connect(x1,y1,x2,y2):
		return (x1-x2)**2+5*y1**2-4*x2**2+y2**2 > 2.5
	def amp(x1,y1,x2,y2):
		return (x1-x2)**2+x2**2 + 1 - y2**2 < 2
	def phase(x1,y1,x2,y2):
		return math.sin(x1)*math.cos(y1)**2 + 1

	joints = query_cppn(wc,5,6,2.1,connect,amp,phase)
	return wc.get_walker()

```

#### G.2  Hopper (from Square Seed) Hopper（来自 Square Seed）

A video of the Sodaracer represented by the code below can be seen at: https://y2u.be/noSPGFX5m3M.

以下代码所代表的 Sodaracer 视频可在以下网址查看：https://y2u.be/noSPGFX5m3M。

```python
def make_square(wc, x0, y0, x1, y1, length):
	j0 = wc.add_joint(x0, y0)
	j1 = wc.add_joint(x0, y1)
	j2 = wc.add_joint(x1, y1)
	j3 = wc.add_joint(x1, y0)

	return j0, j1, j2, j3

def make_walk(n=6):

	wc = walker_creator()

	# the main body is a square
	sides_2_theta = make_square(wc, 0.0, 0.0, 5.6, 9.4, 2.4)
	sides_1_theta = make_square(wc, 0.5, 0.8, 6.5, 13.1, 1.3)
	sides_2_theta += make_square(wc, -0.8, -0.6, 6.7, 13.0, 2.3)
	sides_2_theta += make_square(wc, -0.9, -0.6, 8.4, 12.5, 0.7)
	sides_2_theta += make_square(wc, 0.0, -0.5, 0.2, 12.4, 1.7)
	sides = sides_1_theta + sides_2_theta + sides_1_theta
	center = wc.add_joint(2, 2)

	# connect the square with distance muscles
	for k in range(len(sides)-2):
		wc.add_muscle(sides[k], sides[k+1])
		wc.add_muscle(sides[k+2], sides[k], False, 30.0, 30.0)

	# similarities of the Squares with":
	for k in range(len(sides)-2):
		wc.add_muscle(sides[k], sides[k], True)
		
		for n in range(k, len(sides)):
			wc.add_muscle(sides[k], sides[n], False)
	wc.add_muscle(sides[3], center)
	# the other prongs from the center of the square are active
	wc.add_muscle(sides[2], center, False, 25.0, 25.0-0.7)
	wc.add_muscle(sides[3], center, False, 20.0, 30.0+0.4)
	
	return wc.get_walker()
```

### G.3 Centipede (from Radial Seed) Centipede（来自径向种子）

A video of the Sodaracer represented by the code below can be seen at: https://y2u.be/zhMsPzo22do.

以下代码所代表的 Sodaracer 视频可以在以下网址看到：https://y2u.be/zhMsPzo22do。

```python
import math

def make_circle(wc, cx,cy,radius,num_points,eccentricity=1.4):
	joints = []
 

	tot_ang = math.pi*2.0*eccentricity

	for idx in range(1,num_points):
		x = math.cos(3.14*(idx+num_points)*tot_ang/(num_points))* radius + cx
		y = math.sin(3.14*(idx+num_points)*tot_ang/(num_points))* radius + cy
		joints.append(wc.add_joint(x,y))
 
	return joints
 

def make_walker(num_points=300,rad=3.25,f=3,max_rad=3):
	wc = walker_creator()
 
	cx,cy = (0,0)
	body_size = rad*1.625

	points = make_circle(wc, 0,0,body_size,num_points)
	center = wc.add_joint(cx,cy)

	for k in range(1,num_points-1):
		wc.add_muscle(points[((k%10) - 1) % 10], points[k], False,int(f*k/float(10)), k/10.)
		wc.add_muscle(points[(k%10)], points[k], True, 1, k/10.)

	return wc.get_walker()

```

## H Probing Stage 2 Models 探索第 2 阶段模型

One hope for the models trained in Stage 2 is that they will learn not only to memorize the training data (Python examples of Sodaracers), but also to internalize the underlying structure of the domain (e.g. how in general to mix together springs and masses to create functional Sodarace inventions). This section discusses some preliminary observations of informal experiments that change the training procedure in Stage 2 to explore what the model is capable of learning. In particular, Sodarace examples are augmented with additional comments (either as a pre x or post x) that contain both the Sodaracer’s fitness and its behavior characterization (its width, height, and mass).

对第二阶段训练的模型的一个希望是，它们不仅能学会记忆训练数据（Sodaracers的Python例子），而且能内化该领域的基本结构（例如，一般来说，如何将弹簧和质量混合在一起，创造出功能性的Sodarace发明）。本节讨论了一些非正式实验的初步观察，这些实验改变了第二阶段的训练程序，以探索模型能够学习什么。特别是，Sodarace的例子被增加了额外的评论（作为pre x或post x），这些评论包含了Sodaracer的适用性和它的行为特征（它的宽度、高度和质量）。

The idea is that after training, the model can be asked to predict e.g. the fitness of an unseen invention (if trained with post x comments), or to generate a walker with desired properties (if trained with pre x comments). For example, a prefix-trained model can be conditionally sampled based on a prefix that specifies the desired height, width, and mass of a Sodaracer, to see how reliably samples can match those properties when evaluated in the domain.

这个想法是，在训练之后，模型可以被要求预测例如一个未见过的发明的适用性（如果用post x注释训练），或者生成一个具有所需属性的步行者（如果用pre x注释训练）。例如，一个经过前缀训练的模型可以根据一个前缀有条件地取样，该前缀指定了一个Sodaracer所需的高度、宽度和质量，以观察在该领域中评估时，样本与这些属性的可靠匹配程度。

Preliminary experiments with both prefix and postfix 300M parameter models highlighted that the model was able to make such associations within the training distribution, e.g. when a postfix model was queried with heights, widths, and masses taken from test set examples (held out from the same distribution), it was able to consistently generate a Sodaracer with those properties. It was slightly less reliable when conditioned on fitness, reflecting that this is a much more complicated association (e.g. unlike width and height, fitness depends on the physical dynamics of the generated walker).

用前缀和后缀300M参数模型进行的初步实验表明，该模型能够在训练分布范围内进行这种关联，例如，当后缀模型被问及高度、宽度和取自测试集例子的质量（从同一分布中取出）时，它能够持续生成具有这些属性的Sodaracer。当以适合度为条件时，它的可靠性稍差，反映出这是一个更复杂的关联（例如，与宽度和高度不同，适合度取决于生成的步行者的物理动态）
 
However, when taken out of distribution the model was less robust. For example, a prefix model struggled to targetedly generate Sodaracers within a band of width and height that was deliberately held out from the training set. Interestingly, while it was not reliable in generating Sodaracers of particular held-out widths and heights, samples from the model did in effect cover the holdout area, suggesting that the variation accessible within the model is enough for interpolation or slight extrapolation, which is an important property for enabling continual open-ended elaboration.

然而，当从分布中抽出时，模型就不那么稳健了。例如，一个前缀的模型很难有针对性地生成宽度和高度范围内的Sodaracers，而这个范围是故意从训练集中保留下来的。有趣的是，虽然它在生成特定宽度和高度的Sodaracers方面并不可靠，但来自该模型的样本实际上覆盖了保留区域，这表明模型内可获得的变化足以进行内插或轻微的外推，这对于实现持续的开放式阐述是一个重要属性。

More starkly, a postfix model had very limited ability to predict the fitness of Sodaracers taken from the Radial seed, which was not seen in training (there was a Spearman correlation of only 0.08). One hypothesis that is left to future work to explore, is that larger models, trained with much more generated data, may have more robust performance when taken out-of-distribution. If true, this would support that scaling can benefit open-ended learning, just as it does in unsupervised and supervised learning.

更明显的是，后缀模型预测从Radial种子中提取的Sodaracers适配度的能力非常有限，这在训练中是看不到的（只有0.08的Spearman相关性）。留给未来工作探索的一个假设是，用更多的生成数据训练出来的更大的模型，在采取非分布式的时候可能有更强的性能。如果是真的，这将支持缩放可以使开放式学习受益，就像它在无监督和有监督学习中一样。

A more speculative line of thought emerging from these experiments relates to how Stage 2 structures the knowledge about the domain, which may signiicantly impact the dynamics of how RL in Stage 3 unfolds. That is, by training the model to associate Sodaracers with their properties (through a prefix or postfix), it may be more likely that Stage 3 can smoothly interpolate in the space of those properties, which otherwise the model would have no explicit knowledge about. However, when a prefix-trained model was tested in the interpolation setup of Appendix I, it did not perform any better than those trained without prefixes. While such prefix-training did not have the desired impact, it remains an open question how to include within Stage 2 information that intuitively seems highly-relevant to RL (like tness) in a way that maximally bene ts such RL.

从这些实验中出现的一个更具推测性的思路与第二阶段如何构建关于该领域的知识有关，这可能对第三阶段的RL如何展开的动态产生重大影响。也就是说，通过训练模型将Sodaracers与它们的属性联系起来（通过前缀或后缀），第三阶段可能更有可能在这些属性的空间中顺利插值，否则模型就没有明确的知识。然而，当前缀训练的模型在附录I的插值设置中被测试时，它的表现并不比没有前缀训练的模型好。虽然这样的前缀训练没有产生预期的影响，但如何在第二阶段包括直观上似乎与RL高度相关的信息（如tness），并使这种RL得到最大的好处，仍然是一个开放的问题。

Overall, the conclusion is that (at least with 300M parameter models and the current amount of training data), Stage 2 models demonstrate modest capabilities to learn structure within Sodarace, but are not yet robust when taken out-of-distribution. The implication for open-endedness is unclear (whether or not this poses a problem for future research): For example, it may be that stronger generalization capabilities may more naturally emerge when the existing pipeline (which is mainly a proof of concept) is extended such that Stage 3 is embedded within an open-ended process. Indeed, at least in human processes of innovation, general insight does seemingly often emerge from continual open-ended accumulation of initially-disparate examples of phenomena that are only later unified.

总的来说，结论是（至少在300M参数的模型和目前的训练数据量下），第二阶段的模型在Sodarace内表现出适度的学习结构的能力，但在采取非分布式的时候还不健全。对开放性的影响还不清楚（这是否对未来的研究构成了问题）。例如，当现有的流水线（主要是一个概念证明）被扩展，使第三阶段被嵌入到一个开放式的过程中时，可能会更自然地出现更强的概括能力。事实上，至少在人类的创新过程中，一般的洞察力似乎经常从最初不同的现象的持续的开放式积累中出现，这些现象后来才被统一。

## I Interpolation Experiments 插值实验

This section discusses experiments probing how well the conditional inventor (the product of Stage 3) is able to understand the domain of the invention, by exploring whether the model can appropriately adjust its output in response to structured changes in the environment. That is, adjusting inventions in response to smooth variations in the environment requires a deeper understanding of the structure of the domain, and could potentially enable inventors to generalize beyond the environments observed during training.

本节讨论了通过探索模型是否可以适当地调整其输出以响应环境中的结构化变化来探索有条件发明人（阶段 3 的产品）如何理解发明领域的实验。 也就是说，根据环境中的平滑变化调整发明需要对领域结构有更深入的了解，并且可能使发明者能够在训练期间观察到的环境之外进行概括。

To examine this capability, an environment distribution with smoothly varying features is created, in particular, by varying the heights of tunnel terrains. The motivation for this distribution is the observation that while larger Sodaracers are unable to navigate low tunnels, they tend to locomote more quickly on at terrains. Thus, the model is incentivized to adapt the height of the produced Sodaracer to the height of the tunnel in the terrain, using "taller" Sodaracers that locomote quickly for the taller tunnels, and shorter, slower-moving Sodaracers for the lower tunnels. The ability to achieve such a solution would imply that the model has learned about the underlying structure of the domain, in that it is able to tweak the height of the produced inventions, and has captured this relationship between the height and speed of the Sodaracer. To enable the model to potentially learn a smooth mapping from the height of the tunnel to the produced Sodaracer, the ResNet TEN architecture is employed.

为了研究这种能力，我们创建了一个具有平滑变化特征的环境分布，特别是通过改变隧道地形的高度。这种分布的动机是观察到，虽然较大的Sodaracers无法在低矮的隧道中航行，但它们往往在地形上更快地移动。因此，该模型被激励去适应所生产的Sodaracer的高度，以适应地形中隧道的高度，在较高的隧道中使用 "较高 "的Sodaracer，快速定位，而在较低的隧道中使用较短的、移动较慢的Sodaracer。实现这种解决方案的能力意味着模型已经了解了该领域的基本结构，因为它能够调整所生产的发明的高度，并且已经掌握了Sodaracer的高度和速度之间的这种关系。为了使模型有可能学会从隧道的高度到所生产的Sodaracer的平滑映射，采用了ResNet TEN架构。

In the experiments, however, the model repeatedly converged on solutions outputting the same Sodaracer regardless of the height of the tunnel, i.e. an unconditional solution. Examples of such solutions are shown at https://y2u.be/gt1Z0lnjAuE.

然而，在实验中，无论隧道的高度如何，模型都会反复收敛于输出相同 Sodaracer 的解决方案，即无条件解决方案。 此类解决方案的示例显示在 https://y2u.be/gt1Z0lnjAuE。

These results point towards a subtle characteristic of the invention pipeline introduced in this work. The models are not exhibiting a deep understanding of the domain, finding a local, unconditional optimum that works "reasonably" well on almost all terrains in the distribution. Particularly concerning is that the produced Sodaracer is not able to navigate all terrains in the distribution, highlighting the suboptimality of the learned solution. This property confounds probing the interpolation capabilities of the inventors, and it remains unclear if the invention pipeline can produce complex solutions that are able to smoothly vary the produced inventions in response to smooth variations in the environment. Conversely, the experiments presented in the main body of this document imply that the model is able to produce conditional solutions when no unconditional solutions are sufficient.

这些结果指出了这项工作中引入的发明流水线的一个微妙特点。这些模型没有表现出对该领域的深入理解，找到了一个局部的、无条件的最佳状态，在分布中的几乎所有地形上都能 "合理 "地工作。特别令人关注的是，所产生的Sodaracer不能在分布中的所有地形上导航，突出了所学到的解决方案的次优性。这一特性混淆了对发明者的插值能力的探究，目前仍不清楚发明流水线是否能够产生复杂的解决方案，能够根据环境的平滑变化而平滑地改变所产生的发明。相反，本文主体部分提出的实验意味着，在没有充分的无条件解决方案的情况下，该模型能够产生条件性解决方案。

We speculate that unconditional local optima are simpler and easier to learn using RL methods, such that the models "gravitate" towards them when such solutions exist. However, in future work, the invention pipeline could be deployed in more complex, open-ended processes where unconditional solutions should be rendered insufficient. In such settings, it is conceivable that the pipeline will output conditional inventors that have a deeper understanding of the domain structure, as such solutions will allow the inventors to achieve significantly higher rewards in the domain, negating the concern regarding unconditional solutions.

我们推测，无条件的局部最优值更简单，更容易使用RL方法学习，因此，当这种解决方案存在时，模型会 "倾向于 "它们。然而，在未来的工作中，该发明流水线可以被部署在更复杂、更开放的过程中，在这些过程中无条件的解决方案应该是不够的。在这种情况下，可以想象流水线将输出对领域结构有更深理解的有条件的发明人，因为这种解决方案将使发明人在领域中获得更高的奖励，从而消除对无条件解决方案的担忧。

Another avenue for future research would attempt to make the learning task posed in Stage 3 easier by exploring maximum likelihood learning methods when bootstrapping the conditional inventor (Stages 1 and 2). Here, the assumption is that the exploration task in Stage 3, coupled with the necessity of incorporating the new modality, is quite challenging for RL procedures. A simple approach to this could be to sample from the unconditional LLM multiple times, and use the best-performing samples for each terrain in the distribution as a supervised (terrain-Sodaracer pairs) dataset to ne-tune both the LLM and the TENs. Stage 3 could include terrain-distributions incorporating terrains unseen during Stage 2, encouraging the inventor to further generalize and explore the space of inventions. Looking even further down the line, it is conceivable to replace the MAP-Elites-based ELM procedure of Stage 1 with a POET-style algorithm [40], which would produce a supervised dataset of this form during Stage 1, relieving pipeline designers of the need to hand-specify terrain distributions on which to train the conditional inventor in Stage 2.

未来研究的另一个途径是尝试通过在引导条件发明人时探索最大似然学习方法（阶段 1 和阶段 2），使阶段 3 中提出的学习任务更容易。在这里，假设第 3 阶段的探索任务，加上结合新模式的必要性，对于 RL 程序来说是相当具有挑战性的。一个简单的方法是从无条件 LLM 中多次采样，并使用分布中每个地形的最佳性能样本作为监督（地形-Sodaracer 对）数据集来调整 LLM 和 TEN。第 3 阶段可以包括包含第 2 阶段未见地形的地形分布，鼓励发明人进一步概括和探索发明空间。再往下看，可以想象用 POET 风格的算法 [40] 替换阶段 1 的基于 MAP-Elites 的 ELM 程序，这将在阶段 1 产生这种形式的监督数据集，从而减轻流水线设计人员的需要手动指定在第 2 阶段训练有条件发明人的地形分布。