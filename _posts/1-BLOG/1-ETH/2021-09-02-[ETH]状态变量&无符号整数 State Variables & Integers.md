---
title: 【ETH】状态变量&无符号整数 State Variables & Integers
date: 2021-09-02 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

## 状态变量 State variables
<b><font color="#0099ff">State variables</font></b> are permanently stored in contract storage. This means they're written to the Ethereum blockchain. Think of them like writing to a DB.


<b><font color="#0099ff">状态变量 State variables</font></b> 是被永久地保存在合约中。也就是说它们被写入以太币区块链中. 想象成写入一个数据库。</div>

Example:
```solidity
contract Example {
  // This will be stored permanently in the blockchain
  // 这个无符号整数将会永久的被保存在区块链中
  uint myUnsignedInteger = 100;
}
```
In this example contract, we created a uint called myUnsignedInteger and set it equal to 100.

在上面的例子中，定义 myUnsignedInteger 为 uint 类型，并赋值100。

## 无符号整数:unit Unsigned Integers: uint
The uint data type is an unsigned integer, meaning its value must be non-negative. There's also an int data type for signed integers.

> *Note: In Solidity, uint is actually an alias for uint256, a 256-bit unsigned integer. You can declare uints with less bits — uint8, uint16, uint32, etc.. But in general you want to simply use uint except in specific cases, which we'll talk about in later lessons.*

uint 无符号数据类型， 指其值不能是负数，对于有符号的整数存在名为 int 的数据类型。

> *注: Solidity中， uint 实际上是 uint256代名词， 一个256位的无符号整数。你也可以定义位数少的 uints — uint8， uint16， uint32， 等…… 但一般来讲你愿意使用简单的 uint， 除非在某些特殊情况下，这我们后面会讲*
