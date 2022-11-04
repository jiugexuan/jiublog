---
title: 【ETH】继承 Inheritance
date: 2022-2-20 10:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->


Rather than making one extremely long contract, sometimes it makes sense to split your code logic across multiple contracts to organize the code.

代码过于冗长的时候，最好将代码和逻辑分拆到多个不同的合约中，以便于管理。

One feature of Solidity that makes this more manageable is contract <b><font color="#0099ff">inheritance</font></b>:

有个让 Solidity 的代码易于管理的功能，就是合约 <b><font color="#0099ff">inheritance (继承)</font></b>：

```solidity
contract Doge {
  function catchphrase() public returns (string) {
    return "So Wow CryptoDoge";
  }
}

contract BabyDoge is Doge {
  function anotherCatchphrase() public returns (string) {
    return "Such Moon BabyDoge";
  }
}
}
```

<font color="#800080"><b> BabyDoge </b></font> <b><font color="#0099ff">inherits</font></b> from <font color="#800080"><b> Doge </b></font>. That means if you compile and deploy <font color="#800080"><b> BabyDoge </b></font>, it will have access to both <font color="#800080"><b> catchphrase() </b></font> and <font color="#800080"><b> anotherCatchphrase() </b></font> (and any other public functions we may define on <font color="#800080"><b> Doge </b></font>).

由于 <font color="#800080"><b> BabyDoge </b></font> 是从 <font color="#800080"><b> Doge </b></font> 那里 <b><font color="#0099ff">inherits （继承) </font></b>过来的。 这意味着当你编译和部署了 <font color="#800080"><b> BabyDoge </b></font>，它将可以访问 <font color="#800080"><b> catchphrase() </b></font> 和 <font color="#800080"><b> anotherCatchphrase() </b></font>和其他我们在 <font color="#800080"><b> Doge </b></font>中定义的其他公共函数。

This can be used for logical inheritance (such as with a subclass, a <font color="#800080"><b> Cat </b></font> is an <font color="#800080"><b> Animal </b></font>). But it can also be used simply for organizing your code by grouping similar logic together into different contracts.

这可以用于逻辑继承（比如表达子类的时候，<font color="#800080"><b> Cat </b></font> 是一种 <font color="#800080"><b> Animal </b></font>）。 但也可以简单地将类似的逻辑组合到不同的合约中以组织代码。
