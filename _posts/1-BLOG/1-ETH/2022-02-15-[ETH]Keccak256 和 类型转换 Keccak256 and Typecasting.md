---
title: 【ETH】Keccak256 和 类型转换 Keccak256 and Typecasting
date: 2022-2-15 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->

We want our <font color="#800080"><b> _generateRandomDna </b></font> function to return a (semi) random <font color="#800080"><b> unit </b></font>. How can we accomplish this?

如何让 <font color="#800080"><b> _generateRandomDna </b></font> 函数返回一个全(半) 随机的 <font color="#800080"><b> unit </b></font>?

Ethereum has the hash function <font color="#800080"><b> keccak256 </b></font> built in, which is a version of SHA3. A hash function basically maps an input into a random 256-bit hexadecimal number. A slight change in the input will cause a large change in the hash.

Ethereum 内部有一个散列函数<font color="#800080"><b> keccak256 </b></font>，它用了SHA3版本。一个散列函数基本上就是把一个字符串转换为一个256位的16进制数字。字符串的一个微小变化会引起散列数据极大变化。

It's useful for many purposes in Ethereum, but for right now we're just going to use it for pseudo-random number generation.

这在 Ethereum 中有很多应用，但是现在我们只是用它造一个伪随机数。

Also important, <font color="#800080"><b> keccak256 </b></font> expects a single parameter of type <font color="#800080"><b> bytes </b></font>. This means that we have to "pack" any parameters before calling <font color="#800080"><b> keccak256 </b></font>:

同样重要的是，<font color="#800080"><b> keccak256 </b></font>需要<font color="#800080"><b> bytes </b></font>类型的单个参数。这意味着在调用<font color="#800080"><b> keccak256 </b></font>之前，我们必须“打包”任何参数：

Example:

例子:

```solidity
//6e91ec6b618bb462a4a6ee5aa2cb0e9cf30f7a052bb467b0ba58b8748c00d2e5
keccak256("aaaab");
//b1f078126895a1424524de5321b339ab00408010b7cf0e6ed451514981e58aa9
keccak256("aaaac");
```

As you can see, the returned values are totally different despite only a 1 character change in the input.

显而易见，输入字符串只改变了一个字母，输出就已经天壤之别了。

> *Note: Secure random-number generation in blockchain is a very difficult problem. Our method here is insecure, but since security isn't top priority for our Zombie DNA, it will be good enough for our purposes.<br/>注: 在区块链中安全地产生一个随机数是一个很难的问题， 本例的方法不安全，但是在我们的Zombie DNA算法里不是那么重要，已经很好地满足我们的需要了。*

## 类型转换 Typecasting

Sometimes you need to convert between data types. Take the following example:

有时你需要变换数据类型。例如:

```solidity
uint8 a = 5;
uint b = 6;
// throws an error because a * b returns a uint, not uint8:
// 将会抛出错误，因为 a * b 返回 uint, 而不是 uint8:
uint8 c = a * b;
// we have to typecast b as a uint8 to make it work:
// 我们需要将 b 转换为 uint8:
uint8 c = a * uint8(b);
```

In the above, <font color="#800080"><b> a * b </b></font> returns a  <font color="#800080"><b> unit </b></font>, but we were trying to store it as a uint8, which could cause potential problems. By casting it as a uint8, it works and the compiler won't throw an error.

上面, <font color="#800080"><b> a * b </b></font> 返回类型是  <font color="#800080"><b> unit </b></font>, 但是当我们尝试用 <font color="#800080"><b> unit8 </b></font> 类型接收时, 就会造成潜在的错误。如果把它的数据类型转换为 <font color="#800080"><b> unit8 </b></font>, 就可以了，编译器也不会出错。
