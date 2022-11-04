---
title: 【ETH】引入 Import
date: 2022-2-20 11:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->


When you have multiple files and you want to import one file into another, Solidity uses the <font color="#800080"><b> import </b></font> keyword:

在 Solidity 中，当你有多个文件并且想把一个文件导入另一个文件时，可以使用 <font color="#800080"><b> import </b></font> 语句：

```solidity
import "./someothercontract.sol";

contract newContract is SomeOtherContract {

}
```

So if we had a file named <font color="#800080"><b> someothercontract.sol </b></font> in the same directory as this contract (that's what the <font color="#800080"><b> ./ </b></font> means), it would get imported by the compiler.

这样当我们在合约（contract）目录下有一个名为  <font color="#800080"><b> someothercontract.sol </b></font>的文件（ <font color="#800080"><b> ./ </b></font> 就是同一目录的意思），它就会被编译器导入。
