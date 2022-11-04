---
title: 【ETH】智能协议的永固性 Immutability of Contracts
date: 2022-2-21 12:00:00 +/-0800
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

## 智能协议的永固性 Immutability of Contracts

Up until now, Solidity has looked quite similar to other languages like JavaScript. But there are a number of ways that Ethereum DApps are actually quite different from normal applications.

到现在为止，我们讲的 Solidity 和其他语言没有质的区别，它长得也很像 JavaScript。但是，在有几点以太坊上的 DApp 跟普通的应用程序有着天壤之别。

To start with, after you deploy a contract to Ethereum, it’s <b><font color="#0099ff">immutable</font></b>, which means that it can never be modified or updated again.

第一个例子，在你把智能协议传上以太坊之后，它就变得<b><font color="#0099ff">不可更改</font></b>, 这种永固性意味着你的代码永远不能被调整或更新。

The initial code you deploy to a contract is there to stay, permanently, on the blockchain. This is one reason security is such a huge concern in Solidity. If there's a flaw in your contract code, there's no way for you to patch it later. You would have to tell your users to start using a different smart contract address that has the fix.

你编译的程序会一直，永久的，不可更改的，存在以太坊上。这就是 Solidity 代码的安全性如此重要的一个原因。如果你的智能协议有任何漏洞，即使你发现了也无法补救。你只能让你的用户们放弃这个智能协议，然后转移到一个新的修复后的合约上。

But this is also a feature of smart contracts. The code is law. If you read the code of a smart contract and verify it, you can be sure that every time you call a function it's going to do exactly what the code says it will do. No one can later change that function and give you unexpected results.

但这恰好也是智能合约的一大优势。代码说明一切。如果你去读智能合约的代码，并验证它，你会发现，一旦函数被定义下来，每一次的运行，程序都会严格遵照函数中原有的代码逻辑一丝不苟地执行，完全不用担心函数被人篡改而得到意外的结果。

## 外部依赖关系 External dependencies

In previous chapters, we hard-coded the CryptoKitties contract address into our DApp. But what would happen if the CryptoKitties contract had a bug and someone destroyed all the kitties?It's unlikely, but if this did happen it would render our DApp completely useless — our DApp would point to a hardcoded address that no longer returned any kitties. Our zombies would be unable to feed on kitties, and we'd be unable to modify our contract to fix it.

在之前的篇章中，我们将加密小猫（CryptoKitties）合约的地址硬编码到 DApp 中去了。有没有想过，如果加密小猫出了点问题，比方说，集体消失了会怎么样？ 虽然这种事情几乎不可能发生，但是，如果小猫没了，我们的 DApp 也会随之失效 -- 因为我们在 DApp 的代码中用“硬编码”的方式指定了加密小猫的地址，如果这个根据地址找不到小猫，我们的僵尸也就吃不到小猫了，而按照前面的描述，我们却没法修改合约去应付这个变化！

For this reason, it often makes sense to have functions that will allow you to update key portions of the DApp.

因此，我们不能硬编码，而要采用“函数”，以便于 DApp 的关键部分可以以参数形式修改。

For example, instead of hard coding the CryptoKitties contract address into our DApp, we should probably have a <font color="#800080"><b> setKittyContractAddress </b></font> function that lets us change this address in the future in case something happens to the CryptoKitties contract.

比方说，我们不再一开始就把猎物地址给写入代码，而是写个函数 <font color="#800080"><b> setKittyContractAddress </b></font>, 运行时再设定猎物的地址，这样我们就可以随时去锁定新的猎物，也不用担心加密小猫集体消失了。
