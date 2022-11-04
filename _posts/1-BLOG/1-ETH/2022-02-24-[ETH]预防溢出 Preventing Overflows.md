---
title: 【ETH】预防溢出 Preventing Overflows
date: 2022-2-24 11:00:00 +/-0800
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

Congratulations, that completes our ERC721 and ERC721x implementation!


恭喜你，我们完成了 ERC721 的实现。

That wasn't so tough, was it? A lot of this Ethereum stuff sounds really complicated when you hear people talking about it, so the best way to understand it is to actually go through an implementation of it yourself.

并不是很复杂，对吧？很多类似的以太坊概念，当你只听人们谈论它们的时候，会觉得很复杂。所以最简单的理解方式就是你自己来实现它。

Keep in mind that this is only a minimal implementation. There are extra features we may want to add to our implementation, such as some extra checks to make sure users don't accidentally transfer their token to address <font color="#800080"><b> 0  </b></font> (which is called "burning" a token — basically it's sent to an address that no one has the private key of, essentially making it unrecoverable). Or to put some basic auction logic in the DApp itself. (Can you think of some ways we could implement that?)

不过要记住那只是最简单的实现。还有很多的特性我们也许想加入到我们的实现中来，比如一些额外的检查，来确保用户不会不小心把他们的代币转移给<font color="#800080"><b> 0  </b></font>地址（这被称作 “烧币”, 基本上就是把代币转移到一个谁也没有私钥的地址，让这个代币永远也无法恢复）。 或者在 DApp 中加入一些基本的拍卖逻辑。（你能想出一些实现的方法么？）

But we wanted to keep this lesson manageable, so we went with the most basic implementation. If you want to see an example of a more in-depth implementation, you can take a look at the OpenZeppelin ERC721 contract after this tutorial.

但是为了让我们的课程不至于离题太远，所以我们只专注于一些基础实现。如果你想学习一些更深层次的实现，可以在这个教程结束后，去看看 OpenZeppelin 的 ERC721 合约。

## 合约安全增强: 溢出和下溢 Contract security enhancements: Overflows and Underflows

We're going to look at one major security feature you should be aware of when writing smart contracts: Preventing overflows and underflows.

我们将来学习你在编写智能合约的时候需要注意的一个主要的安全特性：防止溢出和下溢。

What's an <b><font color="#0099ff">overflow</font></b>?

什么是  <b><font color="#0099ff">溢出(overflow)</font></b> ?

Let's say we have a <font color="#800080"><b> uint8 </b></font>, which can only have 8 bits. That means the largest number we can store is binary <font color="#800080"><b> 11111111 </b></font> (or in decimal, 2^8 - 1 = 255).

假设我们有一个 <font color="#800080"><b> uint8 </b></font>, 只能存储8 bit数据。这意味着我们能存储的最大数字就是二进制 <font color="#800080"><b> 11111111 </b></font> (或者说十进制的 2^8 - 1 = 255).

Take a look at the following code. What is <font color="#800080"><b> number </b></font> equal to at the end?

来看看下面的代码。最后 <font color="#800080"><b> number </b></font> 将会是什么值？

```solidity
uint8 number = 255;
number++;
```

In this case, we've caused it to overflow — so <font color="#800080"><b> number </b></font> is counterintuitively now equal to <font color="#800080"><b> 0 </b></font> even though we increased it. (If you add 1 to binary <font color="#800080"><b> 11111111 </b></font>, it resets back to <font color="#800080"><b> 00000000 </b></font>, like a clock going from <font color="#800080"><b> 23:59 </b></font> to <font color="#800080"><b> 00:00 </b></font>).

在这个例子中，我们导致了溢出 — 虽然我们加了1， 但是 <font color="#800080"><b> number </b></font> 出乎意料地等于 <font color="#800080"><b> 0 </b></font>了。 (如果你给二进制 <font color="#800080"><b> 11111111 </b></font> 加1, 它将被重置为 <font color="#800080"><b> 00000000 </b></font>，就像钟表从 <font color="#800080"><b> 23:59 </b></font> 走向 <font color="#800080"><b> 00:00 </b></font>)。

An underflow is similar, where if you subtract <font color="#800080"><b> 1 </b></font> from a <font color="#800080"><b> uint8 </b></font> that equals <font color="#800080"><b> 0 </b></font>, it will now equal <font color="#800080"><b> 255 </b></font> (because <font color="#800080"><b> uints </b></font> are unsigned, and cannot be negative).

下溢(underflow)也类似，如果你从一个等于 0 的 <font color="#800080"><b> uint8 </b></font> 减去 <font color="#800080"><b> 1 </b></font>, 它将变成 <font color="#800080"><b> 255 </b></font> (因为 <font color="#800080"><b> uint </b></font> 是无符号的，其不能等于负数)。

While we're not using <font color="#800080"><b> uint8 </b></font> here, and it seems unlikely that a <font color="#800080"><b> uint256 </b></font> will overflow when incrementing by <font color="#800080"><b> 1 </b></font> each time (2^256 is a really big number), it's still good to put protections in our contract so that our DApp never has unexpected behavior in the future.

虽然我们在这里不使用 <font color="#800080"><b> uint8 </b></font>，而且每次给一个 <font color="#800080"><b> uint256 </b></font> 加 <font color="#800080"><b> 1 </b></font> 也不太可能溢出 (2^256 真的是一个很大的数了)，在我们的合约中添加一些保护机制依然是非常有必要的，以防我们的 DApp 以后出现什么异常情况。

## 使用 SafeMath Using SafeMath

To prevent this, OpenZeppelin has created a  <b><font color="#0099ff"> library </font></b> called SafeMath that prevents these issues by default.

为了防止这些情况，OpenZeppelin 建立了一个叫做 SafeMath 的 <b><font color="#0099ff">库(library)</font></b>，默认情况下可以防止这些问题。

But before we get into that... What's a <b><font color="#0099ff"> library </font></b>?

不过在我们使用之前…… 什么叫做<b><font color="#0099ff"> 库 </font></b>?

A  <b><font color="#0099ff"> library </font></b> is a special type of contract in Solidity. One of the things it is useful for is to attach functions to native data types.

一个<b><font color="#0099ff"> 库 </font></b> 是 Solidity 中一种特殊的合约。其中一个有用的功能是给原始数据类型增加一些方法。

For example, with the SafeMath library, we'll use the syntax <font color="#800080"><b> using SafeMath for uint256 </b></font>. The SafeMath library has 4 functions — <font color="#800080"><b> add </b></font>, <font color="#800080"><b> sub </b></font>, <font color="#800080"><b> mul </b></font>, and <font color="#800080"><b> div </b></font>. And now we can access these functions from <font color="#800080"><b> uint256 </b></font> as follows:

比如，使用 SafeMath 库的时候，我们将使用 <font color="#800080"><b> using SafeMath for uint256 </b></font> 这样的语法。 SafeMath 库有四个方法 — <font color="#800080"><b> add </b></font>，<font color="#800080"><b> sub </b></font> ，<font color="#800080"><b> mul </b></font> ， 以及 <font color="#800080"><b> div </b></font>。现在我们可以这样来让 <font color="#800080"><b> uint256 </b></font> 调用这些方法：

```solidity
using SafeMath for uint256;

uint256 a = 5;
uint256 b = a.add(3); // 5 + 3 = 8
uint256 c = a.mul(2); // 5 * 2 = 10
```

Let's take a look at the code behind SafeMath:

来看看 SafeMath 的部分代码:

```solidity

library SafeMath {

  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
```

First we have the <b><font color="#0099ff"> library </font></b> keyword — libraries are similar to <font color="#800080"><b> contracts </b></font> but with a few differences. For our purposes, libraries allow us to use the <font color="#800080"><b> using </b></font> keyword, which automatically tacks on all of the library's methods to another data type:

首先我们有了 <b><font color="#0099ff"> library </font></b> 关键字 — 库和 <font color="#800080"><b> 合约 </b></font>很相似，但是又有一些不同。 就我们的目的而言，库允许我们使用 <font color="#800080"><b> using </b></font> 关键字，它可以自动把库的所有方法添加给一个数据类型：

```solidity
using SafeMath for uint;
// now we can use these methods on any uint
// 这下我们可以为任何 uint 调用这些方法了
uint test = 2;
test = test.mul(3); // test now equals 6 test 等于 6 了
test = test.add(5); //test now equals 11 test 等于 11 了
```
Note that the <font color="#800080"><b> mul </b></font> and <font color="#800080"><b> add </b></font> functions each require 2 arguments, but when we declare <font color="#800080"><b> using SafeMath for uint </b></font>, the <font color="#800080"><b> uint </b></font> we call the function on (<font color="#800080"><b> test </b></font>) is automatically passed in as the first argument.

注意 <font color="#800080"><b> mul </b></font> 和 <font color="#800080"><b> add </b></font> 其实都需要两个参数。 在我们声明了 <font color="#800080"><b> using SafeMath for uint </b></font> 后，我们用来调用这些方法的 <font color="#800080"><b> uint </b></font> 就自动被作为第一个参数传递进去了(在此例中就是 <font color="#800080"><b> test </b></font>)

Let's look at the code behind <font color="#800080"><b> add</b></font> to see what SafeMath does:

我们来看看 <font color="#800080"><b> add</b></font> 的源代码看 SafeMath 做了什么:

```solidity
function add(uint256 a, uint256 b) internal pure returns (uint256) {
  uint256 c = a + b;
  assert(c >= a);
  return c;
}
```

Basically <font color="#800080"><b> add </b></font> just adds 2 <font color="#800080"><b> uint </b></font> like <font color="#800080"><b> +  </b></font>, but it also contains an <font color="#800080"><b> assert </b></font> statement to make sure the sum is greater than <font color="#800080"><b> a </b></font>. This protects us from overflows.

基本上 <font color="#800080"><b> add </b></font> 只是像 <font color="#800080"><b> +  </b></font>一样对两个 <font color="#800080"><b> uint </b></font> 相加， 但是它用一个 <font color="#800080"><b> assert </b></font> 语句来确保结果大于 <font color="#800080"><b> a </b></font>。这样就防止了溢出。

<font color="#800080"><b> assert </b></font> is similar to <font color="#800080"><b> require </b></font>, where it will throw an error if false. The difference between <font color="#800080"><b> assert </b></font> and <font color="#800080"><b> require </b></font> is that <font color="#800080"><b> require </b></font> will refund the user the rest of their gas when a function fails, whereas <font color="#800080"><b> assert </b></font> will not. So most of the time you want to use <font color="#800080"><b> require </b></font> in your code; <font color="#800080"><b> assert </b></font> is typically used when something has gone horribly wrong with the code (like a <font color="#800080"><b>uint </b></font> overflow).

<font color="#800080"><b> assert </b></font> 和 <font color="#800080"><b> require </b></font> 相似，若结果为否它就会抛出错误。 <font color="#800080"><b> assert </b></font> 和 <font color="#800080"><b> require </b></font> 区别在于，<font color="#800080"><b> require </b></font> 若失败则会返还给用户剩下的 gas， <font color="#800080"><b> assert </b></font> 则不会。所以大部分情况下，你写代码的时候会比较喜欢 <font color="#800080"><b> require </b></font>，<font color="#800080"><b> assert </b></font> 只在代码可能出现严重错误的时候使用，比如 <font color="#800080"><b>uint </b></font> 溢出。

So, simply put, SafeMath's <font color="#800080"><b> add</b></font>, <font color="#800080"><b> sub </b></font>,<font color="#800080"><b> mul </b></font> and <font color="#800080"><b> div </b></font> are functions that do the basic 4 math operations, but throw an error if an overflow or underflow occurs.

所以简而言之， SafeMath 的 <font color="#800080"><b> add</b></font>, <font color="#800080"><b> sub </b></font>,<font color="#800080"><b> mul </b></font> 和 <font color="#800080"><b> div </b></font> 方法只做简单的四则运算，然后在发生溢出或下溢的时候抛出错误。

### 在我们的代码里使用 SafeMath。

To prevent overflows and underflows, we can look for places in our code where we use <font color="#800080"><b> + </b></font>, <font color="#800080"><b> - </b></font>, <font color="#800080"><b> * </b></font>, or <font color="#800080"><b> / </b></font> and replace them with <font color="#800080"><b> add</b></font>,   <font color="#800080"><b> sub </b></font>, <font color="#800080"><b> mul </b></font> ,<font color="#800080"><b> div </b></font> .

为了防止溢出和下溢，我们可以在我们的代码里找 <font color="#800080"><b> + </b></font>， <font color="#800080"><b> - </b></font>， <font color="#800080"><b> * </b></font>， 或 <font color="#800080"><b> / </b></font>，然后替换为 <font color="#800080"><b> add</b></font>, <font color="#800080"><b> sub </b></font>,<font color="#800080"><b> mul </b></font> ,<font color="#800080"><b> div </b></font> .

比如，与其这样做:

```solidity
myUint++;
```

Ex. Instead of doing:

我们这样做：

```solidity
myUint = myUint.add(1);
```

Great, now our ERC721 implementation is safe from overflows & underflows!

太好了，这下我们的 ERC721 实现不会有溢出或者下溢了。

For example:

比如：

```solidity
uint16 winCount;
uint32 level;
uint256 lossCount;
```

We should prevent overflows here as well just to be safe. (It's a good idea in general to just use SafeMath instead of the basic math operations. Maybe in a future version of Solidity these will be implemented by default, but for now we have to take extra security precautions in our code).

我们同样应该在这些地方防止溢出。（通常情况下，总是使用 SafeMath 而不是普通数学运算是个好主意，也许在以后 Solidity 的新版本里这点会被默认实现，但是现在我们得自己在代码里实现这些额外的安全措施）。

However we have a slight problem — <font color="#800080"><b> winCount </b></font> and <font color="#800080"><b> lossCount </b></font> are <font color="#800080"><b> uint16s </b></font>, and <font color="#800080"><b> level </b></font> is a <font color="#800080"><b> uint32 </b></font>. So if we use SafeMath's <font color="#800080"><b> add </b></font> method with these as arguments, it won't actually protect us from overflow since it will convert these types to <font color="#800080"><b> uint256 </b></font>:

不过我们遇到个小问题 — <font color="#800080"><b> winCount </b></font> 和 <font color="#800080"><b> lossCount </b></font> 是 <font color="#800080"><b> uint16 </b></font>， 而 <font color="#800080"><b> level </b></font> 是 <font color="#800080"><b> uint32 </b></font>。 所以如果我们用这些作为参数传入 SafeMath 的 <font color="#800080"><b> add </b></font> 方法。 它实际上并不会防止溢出，因为它会把这些变量都转换成 <font color="#800080"><b> uint256 </b></font>:

```solidity
function add(uint256 a, uint256 b) internal pure returns (uint256) {
  uint256 c = a + b;
  assert(c >= a);
  return c;
}

// If we call `.add` on a `uint8`, it gets converted to a `uint256`.
// So then it won't overflow at 2^8, since 256 is a valid `uint256`.
// 如果我们在`uint8` 上调用 `.add`。它将会被转换成 `uint256`.
// 所以它不会在 2^8 时溢出，因为 256 是一个有效的 `uint256`.
```

This means we're going to need to implement 2 more libraries to prevent overflow/underflows with our <font color="#800080"><b> uint16s </b></font> and <font color="#800080"><b> uint32s </b></font>. We can call them <font color="#800080"><b> SafeMath16 </b></font> and <font color="#800080"><b> SafeMath32 </b></font>.

这就意味着，我们需要再实现两个库来防止 <font color="#800080"><b> uint16 </b></font> 和 <font color="#800080"><b> uint32 </b></font> 溢出或下溢。我们可以将其命名为 <font color="#800080"><b> SafeMath16 </b></font> 和 <font color="#800080"><b> SafeMath32 </b></font>。

The code will be exactly the same as SafeMath, except all instances of uint256 will be replaced with uint32 or uint16.

代码将和 SafeMath 完全相同，除了所有的 <font color="#800080"><b> uint256 </b></font> 实例都将被替换成 <font color="#800080"><b> uint32 </b></font> 或 <font color="#800080"><b> uint16 </b></font>。
