---
title: 【ETH】和合约对话 Talking to Contracts
date: 2022-2-25 10:00:00 +/-0800
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

Now that we've initialized Web3.js with MetaMask's Web3 provider, let's set it up to talk to our smart contract.

现在，我们已经用 MetaMask 的 Web3 提供者初始化了 Web3.js。接下来就让它和我们的智能合约对话吧。

Web3.js will need 2 things to talk to your contract: its <b><font color="#0099ff"> address </font></b> and its <b><font color="#0099ff"> ABI </font></b>.

Web3.js 需要两个东西来和你的合约对话: 它的<b><font color="#0099ff"> 地址 </font></b>和它的<b><font color="#0099ff"> ABI </font></b> 。

## 合约地址 Contract Address

After you finish writing your smart contract, you will compile it and deploy it to Ethereum. We're going to cover deployment in the next lesson, but since that's quite a different process from writing code, we've decided to go out of order and cover Web3.js first.

在你写完了你的智能合约后，你需要编译它并把它部署到以太坊。我们将在下一课中详述部署，因为它和写代码是截然不同的过程，所以我们决定打乱顺序，先来讲 Web3.js。

After you deploy your contract, it gets a fixed address on Ethereum where it will live forever. You'll need to copy this address after deploying in order to talk to your smart contract.

在你部署智能合约以后，它将获得一个以太坊上的永久地址。你需要在部署后复制这个地址以来和你的智能合约对话。

## 合约 ABI Contract ABI

The other thing Web3.js will need to talk to your contract is its <b><font color="#0099ff"> ABI </font></b> .

另一个 Web3.js 为了要和你的智能合约对话而需要的东西是 <b><font color="#0099ff"> ABI </font></b> 。

ABI stands for Application Binary Interface. Basically it's a representation of your contracts' methods in JSON format that tells Web3.js how to format function calls in a way your contract will understand.

ABI 意为应用二进制接口（Application Binary Interface）。 基本上，它是以 JSON 格式表示合约的方法，告诉 Web3.js 如何以合同理解的方式格式化函数调用。

When you compile your contract to deploy to Ethereum , the Solidity compiler will give you the ABI, so you'll need to copy and save this in addition to the contract address.

当你编译你的合约向以太坊部署时， Solidity 编译器会给你 ABI，所以除了合约地址，你还需要把这个也复制下来。

## 实例化 Web3.js Instantiating a Web3.js Contract

Once you have your contract's address and ABI, you can instantiate it in Web3 as follows:

一旦你有了合约的地址和 ABI，你可以像这样来实例化 Web3.js。

```solidity
// Instantiate myContract
// 实例化 myContract
var myContract = new web3js.eth.Contract(myABI, myContractAddress);
```
