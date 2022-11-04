---
title: 【ETH】映射和地址 Mappings and Addresses
date: 2022-2-19 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->

## 地址 Addresses

The Ethereum blockchain is made up of  <b><font color="#0099ff">account</font></b>, which you can think of like bank accounts. An account has a balance of <b><font color="#0099ff">Ether</font></b> (the currency used on the Ethereum blockchain), and you can send and receive Ether payments to other accounts, just like your bank account can wire transfer money to other bank accounts.

以太坊区块链由 <b><font color="#0099ff">账户</font></b> 组成，你可以把它想象成银行账户。一个帐户的余额是 <b><font color="#0099ff">以太</font></b>（在以太坊区块链上使用的币种），你可以和其他帐户之间支付和接受以太币，就像你的银行帐户可以电汇资金到其他银行帐户一样。

Each account has an <font color="#800080"><b> address </b></font>, which you can think of like a bank account number. It's a unique identifier that points to that account, and it looks like this:

每个帐户都有一个“<font color="#800080"><b> 地址 </b></font>”，你可以把它想象成银行账号。这是账户唯一的标识符，它看起来长这样：

```solidity
0x0cE446255506E92DF41614C46F1d6df9Cc969183
```

We'll get into the nitty gritty of addresses in a later lesson, but for now you only need to understand that an address is owned by a specific user (or a smart contract).

我们将在后面介绍地址的细节，现在你只需要了解地址属于特定用户（或智能合约）的。


## 映射 Mapping

In front we looked at <b><font color="#0099ff">structs</font></b> and <b><font color="#0099ff">arrays</font></b>. Mappings are another way of storing organized data in Solidity.

在前面，我们看到了 <b><font color="#0099ff">结构体</font></b> 和 <b><font color="#0099ff">数组</font></b>。<b><font color="#0099ff">映射</font></b> 是另一种在 Solidity 中存储有组织数据的方法。

Defining a <font color="#800080"><b> mapping </b></font> looks like this:

<font color="#800080"><b>映射 </b></font>是这样定义的：

```solidity
// For a financial app, storing a uint that holds the user's account balance:
//对于金融应用程序，将用户的余额保存在一个 uint类型的变量中：
mapping (address => uint) public accountBalance;
// Or could be used to store / lookup usernames based on userId
//或者可以用来通过userId 存储/查找的用户名
mapping (uint => string) userIdToName;
```

A mapping is essentially a key-value store for storing and looking up data. In the first example, the key is an <font color="#800080"><b>address</b></font> and the value is a <font color="#800080"><b>uint </b></font>, and in the second example the key is a <font color="#800080"><b>uint </b></font> and the value a <font color="#800080"><b>string</b></font>.

映射本质上是存储和查找数据所用的键-值对。在第一个例子中，键是一个 <font color="#800080"><b>address</b></font>，值是一个 <font color="#800080"><b>uint </b></font>，在第二个例子中，键是一个<font color="#800080"><b>uint </b></font>，值是一个 <font color="#800080"><b>string</b></font>。
