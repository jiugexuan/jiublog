---
title: 【ETH】处理多返回值 Handling Multiple Return Values
date: 2022-2-21 10:00:00 +/-0800
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


This  <font color="#800080"><b> getKitty </b></font> function is the first example we've seen that returns multiple values. Let's look at how to handle them:

 <font color="#800080"><b> getKitty </b></font>是我们所看到的第一个返回多个值的函数。我们来看看是如何处理的：

```solidity
function multipleReturns() internal returns(uint a, uint b, uint c) {
  return (1, 2, 3);
}

function processMultipleReturns() external {
  uint a;
  uint b;
  uint c;
  // This is how you do multiple assignment:
  // 这样来做批量赋值:
  (a, b, c) = multipleReturns();
}

// 或者如果我们只想返回其中一个变量:
function getLastReturnValue() external {
  uint c;
  // We can just leave the other fields blank:
  // 可以对其他字段留空:
  (,,c) = multipleReturns();
}
```
