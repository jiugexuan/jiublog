---
title: 【ETH】更多关于函数可见性 More on Function Visibility
date: 2022-2-20 13:00:00 +/-0800
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

In addition to <font color="#800080"><b> public </b></font> and <font color="#800080"><b> private </b></font>, Solidity has two more types of visibility for functions: <font color="#800080"><b> internal</b></font> and <font color="#800080"><b> external</b></font> .

除 <font color="#800080"><b> public </b></font> 和 <font color="#800080"><b> private </b></font> 属性之外，Solidity 还使用了另外两个描述函数可见性的修饰词：<font color="#800080"><b> internal（内部）</b></font> 和 <font color="#800080"><b> external（外部）</b></font>。

<font color="#800080"><b> internal</b></font> is the same as <font color="#800080"><b> private </b></font>, except that it's also accessible to contracts that inherit from this contract. (Hey, that sounds like what we want here!).

<font color="#800080"><b> internal</b></font> 和 <font color="#800080"><b> private </b></font> 类似，不过， 如果某个合约继承自其父合约，这个合约即可以访问父合约中定义的“内部”函数。（嘿，这听起来正是我们想要的那样！）。

<font color="#800080"><b> external</b></font> is similar to <font color="#800080"><b> public </b></font>, except that these functions can ONLY be called outside the contract — they can't be called by other functions inside that contract. We'll talk about why you might want to use <font color="#800080"><b> external</b></font> vs <font color="#800080"><b> public </b></font> later.


<font color="#800080"><b> external</b></font> 与public 类似，只不过这些函数只能在合约之外调用 - 它们不能被合约内的其他函数调用。稍后我们将讨论什么时候使用 <font color="#800080"><b> external</b></font> 和 <font color="#800080"><b> public </b></font>。

For declaring  <font color="#800080"><b> internal</b></font> or <font color="#800080"><b> external</b></font> functions, the syntax is the same as <font color="#800080"><b> private </b></font> and <font color="#800080"><b> public </b></font>:

声明函数 <font color="#800080"><b> internal</b></font> 或 <font color="#800080"><b> external</b></font> 类型的语法，与声明 <font color="#800080"><b> private </b></font> 和 <font color="#800080"><b> public </b></font>类 型相同：

```solidity
contract Sandwich {
  uint private sandwichesEaten = 0;

  function eat() internal {
    sandwichesEaten++;
  }
}

contract BLT is Sandwich {
  uint private baconSandwichesEaten = 0;

  function eatWithBacon() public returns (string) {
    baconSandwichesEaten++;
    // We can call this here because it's internal
    // 因为eat() 是internal 的，所以我们能在这里调用
    eat();
  }
}
```
