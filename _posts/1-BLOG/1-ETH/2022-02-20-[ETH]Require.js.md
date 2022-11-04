---
title: 【ETH】Require
date: 2022-2-20 10:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->

<font color="#800080"><b> require </b></font> makes it so that the function will throw an error and stop executing if some condition is not true:

<font color="#800080"><b> require </b></font>使得函数在执行过程中，当不满足某些条件时抛出错误，并停止执行

```solidity
function sayHiToVitalik(string _name) public returns (string) {
  // Compares if _name equals "Vitalik". Throws an error and exits if not true.
  // (Side note: Solidity doesn't have native string comparison, so we
  // compare their keccak256 hashes to see if the strings are equal)
  // 比较 _name 是否等于 "Vitalik". 如果不成立，抛出异常并终止程序
  // (敲黑板: Solidity 并不支持原生的字符串比较, 我们只能通过比较
  // 两字符串的 keccak256 哈希值来进行判断)
  require(keccak256(_name) == keccak256("Vitalik"));
  // If it's true, proceed with the function:
  // 如果返回 true, 运行如下语句
  return "Hi!";
}
```

If you call this function with <font color="#800080"><b> sayHiToVitalik（“Vitalik”） </b></font>, it will return "Hi!". If you call it with any other input, it will throw an error and not execute.

如果你这样调用函数 <font color="#800080"><b> sayHiToVitalik（“Vitalik”） </b></font> ,它会返回“Hi！”。而如果调用的时候使用了其他参数，它则会抛出错误并停止执行。

Thus <font color="#800080"><b> require </b></font> is quite useful for verifying certain conditions that must be true before running a function.

因此，在调用一个函数之前，用 <font color="#800080"><b> require </b></font> 验证前置条件是非常有必要的
