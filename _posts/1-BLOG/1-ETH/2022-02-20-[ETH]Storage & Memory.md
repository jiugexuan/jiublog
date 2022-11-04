---
title: 【ETH】Storage & Memory
date: 2022-2-20 12:00:00 +/-0800
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

In Solidity, there are two locations you can store variables — in <font color="#800080"><b> storage </b></font> and in <font color="#800080"><b> memory </b></font>.

在 Solidity 中，有两个地方可以存储变量 —— <font color="#800080"><b> storage </b></font> 或 <font color="#800080"><b> memory </b></font>。

<b><font color="#0099ff">Storage</font></b> refers to variables stored permanently on the blockchain. <b><font color="#0099ff">Memory</font></b> variables are temporary, and are erased between external function calls to your contract. Think of it like your computer's hard disk vs RAM.

<b><font color="#0099ff">Storage</font></b> 变量是指永久存储在区块链中的变量。 <b><font color="#0099ff">Memory</font></b> 变量则是临时的，当外部函数对某合约调用完成时，内存型变量即被移除。 你可以把它想象成存储在你电脑的硬盘或是RAM中数据的关系。

Most of the time you don't need to use these keywords because Solidity handles them by default. State variables (variables declared outside of functions) are by default <font color="#800080"><b> storage </b></font> and written permanently to the blockchain, while variables declared inside functions are <font color="#800080"><b> memory </b></font> and will disappear when the function call ends.

大多数时候你都用不到这些关键字，默认情况下 Solidity 会自动处理它们。 状态变量（在函数之外声明的变量）默认为<font color="#800080"><b> “存储” </b></font>形式，并永久写入区块链；而在函数内部声明的变量是<font color="#800080"><b> “内存” </b></font>型的，它们函数调用结束后消失。

However, there are times when you do need to use these keywords, namely when dealing with <b><font color="#0099ff">structs</font></b> and <b><font color="#0099ff">arrays</font></b> within functions:

然而也有一些情况下，你需要手动声明存储类型，主要用于处理函数内的  <b><font color="#0099ff">结构体</font></b> 和 <b><font color="#0099ff">数组</font></b> 时：

```solidity
contract SandwichFactory {
  struct Sandwich {
    string name;
    string status;
  }

  Sandwich[] sandwiches;

  function eatSandwich(uint _index) public {
    // Sandwich mySandwich = sandwiches[_index];

    // ^ Seems pretty straightforward, but solidity will give you a warning
    // telling you that you should explicitly declare `storage` or `memory` here.
    // ^ 看上去很直接，不过 Solidity 将会给出警告
    // 告诉你应该明确在这里定义 `storage` 或者 `memory`。

    // So instead, you should declare with the `storage` keyword, like:
    // 所以你应该明确定义 `storage`:
    Sandwich storage mySandwich = sandwiches[_index];
    // ...in which case `mySandwich` is a pointer to `sandwiches[_index]`
    // in storage, and...
    // ...这样 `mySandwich` 是指向 `sandwiches[_index]`的指针
    // 在存储里，另外...
    mySandwich.status = "Eaten!";
    // ...this will permanently change `sandwiches[_index]` on the blockchain.
    // ...这将永久把 `sandwiches[_index]` 变为区块链上的存储

    // If you just want a copy, you can use `memory`:
    // 如果你只想要一个副本，可以使用`memory`:
    Sandwich memory anotherSandwich = sandwiches[_index + 1];
    // ...in which case `anotherSandwich` will simply be a copy of the
    // data in memory, and...
    // ...这样 `anotherSandwich` 就仅仅是一个内存里的副本了
    // 另外
    anotherSandwich.status = "Eaten!";
    // ...will just modify the temporary variable and have no effect
    // on `sandwiches[_index + 1]`. But you can do this:
    // ...将仅仅修改临时变量，对 `sandwiches[_index + 1]` 没有任何影响
    // 不过你可以这样做:
    sandwiches[_index + 1] = anotherSandwich;
    // ...if you want to copy the changes back into blockchain storage.
    // ...如果你想把副本的改动保存回区块链存储
  }
}
```

Don't worry if you don't fully understand when to use which one yet — the Solidity compiler will also give you warnings to let you know when you should be using one of these keywords.

如果你还没有完全理解究竟应该使用哪一个，也不用担心 当你不得不使用到这些关键字的时候，Solidity 编译器也发警示提醒你的。

For now, it's enough to understand that there are cases where you'll need to explicitly declare <font color="#800080"><b> storage </b></font> or <font color="#800080"><b> memory </b></font>!

现在，只要知道在某些场合下也需要你显式地声明 <font color="#800080"><b> storage </b></font> 或 <font color="#800080"><b> memory </b></font>就够了！
