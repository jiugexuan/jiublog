---
title: 【ETH】注释 Comments
date: 2022-2-24 12:00:00 +/-0800
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

Next, we'll look at how to deploy the code to Ethereum, and how to interact with it with <b><font color="#0099ff">Web3.js</font></b>.

在以后的课程中，我们将学习如何将游戏部署到以太坊，以及如何和 <b><font color="#0099ff">Web3.js</font></b> 交互。

我们来谈谈如何 给你的代码添加注释.

Let's talk about commenting your code.

## 注释语法 Syntax for comments

Commenting in Solidity is just like JavaScript. You've already seen some examples of single line comments throughout the CryptoZombies lessons:

Solidity 里的注释和 JavaScript 相同。在我们的课程中你已经看到了不少单行注释了：

```solidity
// This is a single-line comment. It's kind of like a note to self (or to others)
// 这是一个单行注释，可以理解为给自己或者别人看的笔记
```

Just add double <font color="#800080"><b> // </b></font> anywhere and you're commenting. It's so easy that you should do it all the time.

只要在任何地方添加一个 <font color="#800080"><b> // </b></font> 就意味着你在注释。如此简单所以你应该经常这么做。

But I hear you — sometimes a single line is not enough. You are born a writer, after all!

不过我们也知道你的想法：有时候单行注释是不够的。毕竟你生来是个作家。

Thus we also have multi-line comments:

所以我们有了多行注释：
```solidity
contract CryptoZombies {
  /*
  This is a multi-lined comment. Know that this is still the beginning of Blockchain development.
  */

  这是一个多行注释。要知道这依然只是区块链开发的开始而已，虽然我们已经走了很远

  */
}
```
In particular, it's good practice to comment your code to explain the expected behavior of every function in your contract. This way another developer (or you, after a 6 month hiatus from a project!) can quickly skim and understand at a high level what your code does without having to read the code itself.

特别是，最好为你合约中每个方法添加注释来解释它的预期行为。这样其他开发者（或者你自己，在6个月以后再回到这个项目中）可以很快地理解你的代码而不需要逐行阅读所有代码。

The standard in the Solidity community is to use a format called <b><font color="#0099ff">natspec</font></b>, which looks like this:

Solidity 社区所使用的一个标准是使用一种被称作 <b><font color="#0099ff">natspec</font></b> 的格式，看起来像这样：

```solidity
/// @title A contract for basic math operations
/// @author jiugexuan
/// @notice For now, this contract just adds a multiply function
// @title 一个简单的基础运算合约
/// @author jiugexuan
/// @notice 现在，这个合约只添加一个乘法
contract Math {
  /// @notice Multiplies 2 numbers together
  /// @param x the first uint.
  /// @param y the second uint.
  /// @return z the product of (x * y)
  /// @dev This function does not currently check for overflows
  /// @notice 两个数相乘
  /// @param x 第一个 uint
  /// @param y  第二个 uint
  /// @return z  (x * y) 的结果
  /// @dev 现在这个方法不检查溢出
  function multiply(uint x, uint y) returns (uint z) {
    // This is just a normal comment, and won't get picked up by natspec
    // 这只是个普通的注释，不会被 natspec 解释
    z = x * y;
  }
}
```

<font color="#800080"><b> @title </b></font> and <font color="#800080"><b> @author </b></font> are straightforward.

<font color="#800080"><b> @title（标题）</b></font>和<font color="#800080"><b> @author （作者） </b></font>很直接了.

<font color="#800080"><b> @notice </b></font> explains to a user what the contract / function does. <font color="#800080"><b> @dev </b></font> is for explaining extra details to developers.

<font color="#800080"><b> @notice （须知）</b></font> 向 <b>用户</b> 解释这个方法或者合约是做什么的。<font color="#800080"><b> @dev （开发者）</b></font> 是向开发者解释更多的细节。

<font color="#800080"><b> @param </b></font> and <font color="#800080"><b> @return </b></font> are for describing what each parameter and return value of a function are for.

<font color="#800080"><b> @param （参数） </b></font>和<font color="#800080"><b> @return （返回） </b></font>用来描述这个方法需要传入什么参数以及返回什么值

Note that you don't always have to use all of these tags for every function — all tags are optional. But at the very least, leave a <font color="#800080"><b> @dev </b></font> note explaining what each function does.

注意你并不需要每次都用上所有的标签，它们都是可选的。不过最少，写下一个 <font color="#800080"><b> @dev </b></font> 注释来解释每个方法是做什么的。
