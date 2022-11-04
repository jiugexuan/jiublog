---
title: 【ETH】将结构体作为参数传入 Passing structs as arguments
date: 2022-2-22 12:00:00 +/-0800
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

You can pass a storage pointer to a struct as an argument to a <font color="#800080"><b> private </b></font> or <font color="#800080"><b> internal </b></font> function.

由于结构体的存储指针可以以参数的方式传递给一个 <font color="#800080"><b> private </b></font> 或 <font color="#800080"><b> internal </b></font> 的函数，因此结构体可以在多个函数之间相互传递。

The syntax looks like this:

遵循这样的语法：

```solidity
function _doStuff(Thing storage _thing) internal {

}
```
