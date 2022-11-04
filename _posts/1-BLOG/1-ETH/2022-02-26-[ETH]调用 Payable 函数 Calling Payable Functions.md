---
title: 【ETH】调用 Payable 函数 Calling Payable Functions
date: 2022-2-26 08:00:00 +/-0800
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

The logic for <font color="#800080"><b> attack </b></font>,<font color="#800080"><b> changeName </b></font> , and <font color="#800080"><b> changeDna </b></font> will be extremely similar, so they're trivial to implement and we won't spend time coding them in this lesson.

<font color="#800080"><b> attack </b></font>, <font color="#800080"><b> changeName </b></font>, 以及 <font color="#800080"><b> changeDna </b></font> 的逻辑将非常雷同，所以本课将不会花时间在上面。

> *In fact, there's already a lot of repetitive logic in each of these function calls, so it would probably make sense to refactor and put the common code in its own function. (And use a templating system for the <font color="#800080"><b> txStatus </b></font> messages — already we're seeing how much cleaner things would be with a framework like Vue.js!)<br/>实际上，在调用这些函数的时候已经有了非常多的重复逻辑。所以最好是重构代码把相同的代码写成一个函数。（并对<font color="#800080"><b> txStatus </b></font>使用模板系统——我们已经看到用类似 Vue.js 类的框架是多么整洁）*

Let's look at another type of function that requires special treatment in Web3.js — <font color="#800080"><b> payable </b></font> functions.

我们来看看另外一种 Web3.js 中需要特殊对待的函数 —  <font color="#800080"><b> payable </b></font> 函数。

## 例子 example:

```solidity
function levelUp(uint _zombieId) external payable {
  require(msg.value == levelUpFee);
  characters[_zombieId].level++;
}
```

The way to send Ether along with a function is simple, with one caveat: we need to specify how much to send in <font color="#800080"><b> wei </b></font>, not Ether.

和函数一起发送以太非常简单，只有一点需要注意： 我们需要指定发送多少 <font color="#800080"><b> wei </b></font>，而不是以太。

## 什么是 Wei? What's a Wei?

A <font color="#800080"><b> wei </b></font> is the smallest sub-unit of Ether — there are 10^18 <font color="#800080"><b> wei </b></font> in one <font color="#800080"><b> ether </b></font>.

一个 <font color="#800080"><b> wei </b></font> 是以太的最小单位 — 1 <font color="#800080"><b> ether </b></font> 等于 10^18 <font color="#800080"><b> wei </b></font>

That's a lot of zeroes to count — but luckily Web3.js has a conversion utility that does this for us.

太多0要数了，不过幸运的是 Web3.js 有一个转换工具来帮我们做这件事：

```solidity
// This will convert 1 ETH to Wei
// 把 1 ETH 转换成 Wei
web3js.utils.toWei("1", "ether");
```
In our DApp, we set <font color="#800080"><b> levelUpFee = 0.001 ether </b></font>, so when we call our <font color="#800080"><b> levelUp </b></font> function, we can make the user send <font color="#800080"><b> 0.001 </b></font> Ether along with it using the following code:

在我们的 DApp 里， 我们设置了 <font color="#800080"><b> levelUpFee = 0.001 ether </b></font>，所以调用 <font color="#800080"><b> levelUp </b></font> 方法的时候，我们可以让用户用以下的代码同时发送 <font color="#800080"><b> 0.001 </b></font> 以太:

```solidity
cryptoCharacters.methods.levelUp(zombieId)
.send({ from: userAccount, value: web3js.utils.toWei("0.001","ether") })
```
