---
title: 【ETH】循环 For Loops
date: 2022-2-22 15:00:00 +/-0800
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

The syntax of <font color="#800080"><b> for </b></font> loops in Solidity is similar to JavaScript.

<font color="#800080"><b> for </b></font>循环的语法在 Solidity 和 JavaScript 中类似。

Let's look at an example where we want to make an array of even numbers:

来看一个创建偶数数组的例子：

```solidity
function getEvens() pure external returns(uint[]) {
  uint[] memory evens = new uint[](5);
  // Keep track of the index in the new array:
  // 在新数组中记录序列号
  uint counter = 0;
  // Iterate 1 through 10 with a for loop:
  // 在循环从1迭代到10：
  for (uint i = 1; i <= 10; i++) {
    // 如果 `i` 是偶数...
    if (i % 2 == 0) {
      // Add it to our array
      // 把它加入偶数数组
      evens[counter] = i;
      // Increment counter to the next empty index in `evens`:
      //索引加一， 指向下一个空的‘even’
      counter++;
    }
  }
  return evens;
}
```

This function will return an array with the contents <font color="#800080"><b> [2,4,6,8,10] </b></font>.

这个函数将返回一个形为 <font color="#800080"><b> [2,4,6,8,10] </b></font> 的数组。
