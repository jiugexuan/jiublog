---
title: 【ETH】随机数 Random Numbers
date: 2022-2-23 09:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **
```solidity
```

--->

How do we generate random numbers in Solidity?

我们在 Solidity 里如何生成随机数呢？

The real answer here is, you can't. Well, at least you can't do it safely.

真正的答案是你不能，或者最起码，你无法安全地做到这一点。

Let's look at why.

我们来看看为什么

## 用 keccak256 来制造随机数。 Random number generation via keccak256

The best source of randomness we have in Solidity is the <font color="#800080"><b> keccak256 </b></font> hash function.

Solidity 中最好的随机数生成器是<font color="#800080"><b> keccak256 </b></font> 哈希函数.

We could do something like the following to generate a random number:

我们可以这样来生成一些随机数

```solidity
// Generate a random number between 1 and 100:
// 生成一个0到100的随机数:
uint randNonce = 0;
uint random = uint(keccak256(now, msg.sender, randNonce)) % 100;
randNonce++;
uint random2 = uint(keccak256(now, msg.sender, randNonce)) % 100;
```

What this would do is take the timestamp of <font color="#800080"><b> now </b></font>, the <font color="#800080"><b> msg.sender </b></font>, and an incrementing <font color="#800080"><b> nonce </b></font> (a number that is only ever used once, so we don't run the same hash function with the same input parameters twice).

这个方法首先拿到 <font color="#800080"><b> now </b></font> 的时间戳、 <font color="#800080"><b> msg.sender </b></font>、 以及一个自增数 <font color="#800080"><b> nonce </b></font> （一个仅会被使用一次的数，这样我们就不会对相同的输入值调用一次以上哈希函数了）。

It would then "pack" the inputs and use <font color="#800080"><b> keccak </b></font> to convert them to a random hash. Next, it would convert that hash to a <font color="#800080"><b> uint </b></font>, and then use <font color="#800080"><b> % 100 </b></font> to take only the last 2 digits. This will give us a totally random number between 0 and 99.

然后利用 <font color="#800080"><b> keccak </b></font> 把输入的值转变为一个哈希值, 再将哈希值转换为 <font color="#800080"><b> uint </b></font>, 然后利用 <font color="#800080"><b> % 100 </b></font> 来取最后两位, 就生成了一个0到100之间随机数了。

<b>This method is vulnerable to attack by a dishonest node. 这个方法很容易被不诚实的节点攻击</b>

In Ethereum, when you call a function on a contract, you broadcast it to a node or nodes on the network as a <b><font color="#0099ff">transaction</font></b>. The nodes on the network then collect a bunch of transactions, try to be the first to solve a computationally-intensive mathematical problem as a "Proof of Work", and then publish that group of transactions along with their Proof of Work (PoW) as a <b><font color="#0099ff">block</font></b> to the rest of the network.

在以太坊上, 当你在和一个合约上调用函数的时候, 你会把它广播给一个节点或者在网络上的 <b><font color="#0099ff">transaction</font></b> 节点们。 网络上的节点将收集很多事务, 试着成为第一个解决计算密集型数学问题的人，作为“工作证明”，然后将“工作证明”(Proof of Work, PoW)和事务一起作为一个 <b><font color="#0099ff">block</font></b> 发布在网络上。

Once a node has solved the PoW, the other nodes stop trying to solve the PoW, verify that the other node's list of transactions are valid, and then accept the block and move on to trying to solve the next block.

一旦一个节点解决了一个PoW, 其他节点就会停止尝试解决这个 PoW, 并验证其他节点的事务列表是有效的，然后接受这个节点转而尝试解决下一个节点。

This makes our random number function exploitable.

这就让我们的随机数函数变得可利用了

Let's say we had a coin flip contract — heads you double your money, tails you lose everything. Let's say it used the above random function to determine heads or tails. (<font color="#800080"><b> random >= 50 </b></font> is heads, <font color="#800080"><b> random < 50 </b></font> is tails).

我们假设我们有一个硬币翻转合约——正面你赢双倍钱，反面你输掉所有的钱。假如它使用上面的方法来决定是正面还是反面 (<font color="#800080"><b> random >= 50 </b></font> 算正面, <font color="#800080"><b> random < 50 </b></font> 算反面)。

If I were running a node, I could publish a transaction only to my own node and not share it. I could then run the coin flip function to see if I won — and if I lost, choose not to include that transaction in the next block I'm solving. I could keep doing this indefinitely until I finally won the coin flip and solved the next block, and profit.

如果我正运行一个节点，我可以 只对我自己的节点 发布一个事务，且不分享它。 我可以运行硬币翻转方法来偷窥我的输赢 — 如果我输了，我就不把这个事务包含进我要解决的下一个区块中去。我可以一直运行这个方法，直到我赢得了硬币翻转并解决了下一个区块，然后获利。

<b> So how do we generate random numbers safely in Ethereum?  所以我们该如何在以太坊上安全地生成随机数呢 </b>

Because the entire contents of the blockchain are visible to all participants, this is a hard problem, and its solution is beyond the scope of this tutorial. You can read this [StackOverflow](https://ethereum.stackexchange.com/questions/191/how-can-i-securely-generate-a-random-number-in-my-smart-contract)  thread for some ideas. One idea would be to use an <b><font color="#0099ff">oracle</font></b> to access a random number function from outside of the Ethereum blockchain.

因为区块链的全部内容对所有参与者来说是透明的， 这就让这个问题变得很难，它的解决方法不在本文讨论范围，你可以阅读 这个 [StackOverflow](https://ethereum.stackexchange.com/questions/191/how-can-i-securely-generate-a-random-number-in-my-smart-contract) 上的讨论 来获得一些主意。 一个方法是利用 <b><font color="#0099ff">oracle</font></b> 来访问以太坊区块链之外的随机数函数。

Of course, since tens of thousands of Ethereum nodes on the network are competing to solve the next block, my odds of solving the next block are extremely low. It would take me a lot of time or computing resources to exploit this profitably — but if the reward were high enough (like if I could bet $100,000,000 on the coin flip function), it would be worth it for me to attack.

当然， 因为网络上成千上万的以太坊节点都在竞争解决下一个区块，我能成功解决下一个区块的几率非常之低。 这将花费我们巨大的计算资源来开发这个获利方法 — 但是如果奖励异常地高(比如我可以在硬币翻转函数中赢得 1个亿)， 那就很值得去攻击了。

So while this random number generation is NOT secure on Ethereum, in practice unless our random function has a lot of money on the line, the users  likely won't have enough resources to attack it.

所以尽管这个方法在以太坊上不安全，在实际中，除非我们的随机函数有一大笔钱在上面，用户一般是没有足够的资源去攻击的。
