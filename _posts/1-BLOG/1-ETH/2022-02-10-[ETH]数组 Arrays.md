---
title: 【ETH】数组 Structs
date: 2022-2-10 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

## 数组 Structs

When you want a collection of something, you can use an <b><font color="#0099ff">array</font></b>. There are two types of arrays in Solidity: <b><font color="#0099ff">fixed arrats</font></b> and <b><font color="#0099ff">dynamic arrays</font></b>:

如果你想建立一个集合，可以用 <b><font color="#0099ff">数组</font></b> 这样的数据类型. Solidity 支持两种数组: <b><font color="#0099ff">静态数组</font></b> 和 <b><font color="#0099ff">动态数组 </font></b>:

```solidity
// Array with a fixed length of 2 elements:
// 固定长度为2的静态数组:
uint[2] fixedArray;
// another fixed Array, can contain 5 strings:
// 固定长度为5的string类型的静态数组:
string[5] stringArray;
// a dynamic Array - has no fixed size, can keep growing:
// 动态数组，长度不固定，可以动态添加元素:
uint[] dynamicArray;
```
You can also create an array of <b><font color="#0099ff">structs</font></b>. Using the previous chapter's <font color=purple><b> Person </b></font> struct:

你也可以建立一个 <b><font color="#0099ff">结构体类型</font></b> 的数组 例如，上一章提到的 <font color=purple><b> Person </b></font> :

```solidity
Person[] people; // 这是动态数组，我们可以不断添加元素
// dynamic Array, we can keep adding to it
```

Remember that state variables are stored permanently in the blockchain? So creating a dynamic array of structs like this can be useful for storing structured data in your contract, kind of like a database.

记住：状态变量被永久保存在区块链中。所以在你的合约中创建动态数组来保存成结构的数据是非常有意义的。

## 公共数组

You can declare an array as <font color=purple><b> public </b></font>, and Solidity will automatically create a <b><font color="#0099ff">getter</font></b> method for it. The syntax looks like:

你可以定义 <font color=purple><b> public </b></font> 数组, Solidity 会自动创建 <b><font color="#0099ff">getter</font></b> 方法. 语法如下:

```solidity
Person[] public people;
```
Other contracts would then be able to read from, but not write to, this array. So this is a useful pattern for storing public data in your contract.

其它的合约可以从这个数组读取数据（但不能写入数据），所以这在合约中是一个有用的保存公共数据的模式。
