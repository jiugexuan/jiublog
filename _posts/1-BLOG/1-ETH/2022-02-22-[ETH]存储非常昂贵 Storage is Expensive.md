---
title: 【ETH】存储非常昂贵 Storage is Expensive
date: 2022-2-22 14:00:00 +/-0800
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

## 存储非常昂贵 Storage is Expensive

One of the more expensive operations in Solidity is using <font color="#800080"><b> storage </b></font> — particularly writes.

Solidity 使用<font color="#800080"><b> storage(存储) </b></font>是相当昂贵的，”写入“操作尤其贵。

This is because every time you write or change a piece of data, it’s written permanently to the blockchain. Forever! Thousands of nodes across the world need to store that data on their hard drives, and this amount of data keeps growing over time as the blockchain grows. So there's a cost to doing that.

这是因为，无论是写入还是更改一段数据， 这都将永久性地写入区块链。”永久性“啊！需要在全球数千个节点的硬盘上存入这些数据，随着区块链的增长，拷贝份数更多，存储量也就越大。这是需要成本的！

In order to keep costs down, you want to avoid writing data to storage except when absolutely necessary. Sometimes this involves seemingly inefficient programming logic — like rebuilding an array in <font color="#800080"><b> memory </b></font> every time a function is called instead of simply saving that array in a variable for quick lookups.

为了降低成本，不到万不得已，避免将数据写入存储。这也会导致效率低下的编程逻辑 - 比如每次调用一个函数，都需要在 <font color="#800080"><b> memory(内存) </b></font> 中重建一个数组，而不是简单地将上次计算的数组给存储下来以便快速查找。

In most programming languages, looping over large data sets is expensive. But in Solidity, this is way cheaper than using <font color="#800080"><b> storage </b></font> if it's in an <font color="#800080"><b> external view </b></font> function, since <font color="#800080"><b> view </b></font> functions don't cost your users any gas. (And gas costs your users real money!).

在大多数编程语言中，遍历大数据集合都是昂贵的。但是在 Solidity 中，使用一个标记了<font color="#800080"><b> external view </b></font>的函数，遍历比 <font color="#800080"><b> storage </b></font> 要便宜太多，因为 <font color="#800080"><b> view </b></font> 函数不会产生任何花销。 （gas可是真金白银啊！）。

We'll go over <font color="#800080"><b> for </b></font> loops in the next chapter, but first, let's go over how to declare arrays in memory.

我们将在下一章讨论<font color="#800080"><b> for </b></font>循环，现在我们来看一下看如何如何在内存中声明数组。

## 在内存中声明数组 Declaring arrays in memory

You can use the <font color="#800080"><b> memory </b></font> keyword with arrays to create a new array inside a function without needing to write anything to storage. The array will only exist until the end of the function call, and this is a lot cheaper gas-wise than updating an array in <font color="#800080"><b> storage </b></font> — free if it's a <font color="#800080"><b> view </b></font> function called externally.

在数组后面加上 <font color="#800080"><b> memory </b></font>关键字， 表明这个数组是仅仅在内存中创建，不需要写入外部存储，并且在函数调用结束时它就解散了。与在程序结束时把数据保存进 <font color="#800080"><b> storage </b></font> 的做法相比，内存运算可以大大节省gas开销 -- 把这数组放在<font color="#800080"><b> view </b></font>里用，完全不用花钱。

Here's how to declare an array in memory:

以下是申明一个内存数组的例子：

```solidity
function getArray() external pure returns(uint[]) {

  // Instantiate a new array in memory with a length of 3
  // 初始化一个长度为3的内存数组
  uint[] memory values = new uint[](3);

  // Put some values to it
  // 赋值
  values.push(1);
  values.push(2);
  values.push(3);

  // 返回数组
  return values;
}
```

This is a trivial example just to show you the syntax, but in the next chapter we'll look at combining this with <font color="#800080"><b> for </b></font> loops for real use-cases.

这个小例子展示了一些语法规则，下一章中，我们将通过一个实际用例，展示它和 <font color="#800080"><b> for </b></font> 循环结合的做法。

> *Note: memory arrays must be created with a length argument (in this example, <font color="#800080"><b> 3 </b></font>). They currently cannot be resized like storage arrays can with <font color="#800080"><b> array.push() </b></font>, although this may be changed in a future version of Solidity.<br/>注意：内存数组 必须 用长度参数（在本例中为<font color="#800080"><b> 3 </b></font>）创建。目前不支持 <font color="#800080"><b> array.push() </b></font>之类的方法调整数组大小，在未来的版本可能会支持长度修改。*
