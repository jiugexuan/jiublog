---
title: 【ETH】函数的更多属性  More on Functions
date: 2022-2-14 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->

In this chapter, we're going to learn about function return values, and function modifiers.

本章介绍函数的返回值和修饰符。

## 返回值 Return Values ###

In this chapter, we're going to learn about function return values, and function modifiers.

要想函数返回一个数值，按如下定义：


```solidity
string greeting = "What's up ";

function sayHello() public returns (string) {
  return greeting;
}
```

In Solidity, the function declaration contains the type of the return value (in this case <font color=purple><b> string </b></font>).

Solidity 里，函数的定义里可包含返回值的数据类型(如本例中 <font color="#800080"><b> string </b></font>)。

## 函数的修饰符 Function modifiers ###
The above function doesn't actually change state in Solidity — e.g. it doesn't change any values or write anything.

上面的函数实际上没有改变 Solidity 里的状态，即，它没有改变任何值或者写任何东西。

So in this case we could declare it as a <b><font color="#0099ff">view</font></b> function, meaning it's only viewing the data but not modifying it:

这种情况下我们可以把函数定义为 <b><font color="#0099ff">view</font></b>, 意味着它只能读取数据不能更改数据:


```solidity
function sayHello() public view returns (string) {}
```

Solidity also contains <b><font color="#0099ff">pure</font></b>  functions, which means you're not even accessing any data in the app. Consider the following:

Solidity 还支持 <b><font color="#0099ff">pure</font></b>  函数, 表明这个函数甚至都不访问应用里的数据，例如：

```solidity
function _multiply(uint a, uint b) private pure returns (uint) {
  return a * b;
}
```

This function doesn't even read from the state of the app — its return value depends only on its function parameters. So in this case we would declare the function as <b><font color="#0099ff">pure</font></b>.

这个函数甚至都不读取应用里的状态 — 它的返回值完全取决于它的输入参数，在这种情况下我们把函数定义为 <b><font color="#0099ff">pure</font></b>.

> *Note: It may be hard to remember when to mark functions as pure/view. Luckily the Solidity compiler is good about issuing warnings to let you know when you should use one of these modifiers.<br/>注：可能很难记住何时把函数标记为 pure/view。 幸运的是， Solidity 编辑器会给出提示，提醒你使用这些修饰符。*
