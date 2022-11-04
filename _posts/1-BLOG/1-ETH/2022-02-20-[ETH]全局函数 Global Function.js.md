---
title: 【ETH】全局函数 Global Function
date: 2022-2-20 09:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

<!---

<font color="#800080"><b> 私有 </b></font>
<b><font color="#0099ff">结构体类型</font></b>
> **

--->


## Msg.sender

In Solidity, there are certain global variables that are available to all functions. One of these is <font color="#800080"><b> msg.sender </b></font>, which refers to the <font color="#800080"><b> address </b></font> of the person (or smart contract) who called the current function.

在 Solidity 中，有一些全局变量可以被所有函数调用。 其中一个就是 <font color="#800080"><b> msg.sender </b></font>，它指的是当前调用者（或智能合约）的 <font color="#800080"><b> address </b></font>。

> *Note: In Solidity, function execution always needs to start with an external caller. A contract will just sit on the blockchain doing nothing until someone calls one of its functions. So there will always be a <font color="#800080"><b> msg.sender </b></font>.<br/>注意：在 Solidity 中，功能执行始终需要从外部调用者开始。 一个合约只会在区块链上什么也不做，除非有人调用其中的函数。所以 <font color="#800080"><b> msg.sender </b></font>总是存在的。*

Here's an example of using <font color="#800080"><b> msg.sender </b></font> and updating a <font color="#800080"><b> mapping </b></font>:

以下是使用 <font color="#800080"><b> msg.sender </b></font> 来更新  <font color="#800080"><b> mapping </b></font>的例子：

```solidity
mapping (address => uint) favoriteNumber;

function setMyNumber(uint _myNumber) public {
  // Update our `favoriteNumber` mapping to store `_myNumber` under `msg.sender`
  // 更新我们的 `favoriteNumber` 映射来将 `_myNumber`存储在 `msg.sender`名下
  favoriteNumber[msg.sender] = _myNumber;
  // 存储数据至映射的方法和将数据存储在数组相似
  // ^ The syntax for storing data in a mapping is just like with arrays
}

function whatIsMyNumber() public view returns (uint) {
  // Retrieve the value stored in the sender's address
  // 拿到存储在调用者地址名下的值
  // Will be `0` if the sender hasn't called `setMyNumber` yet
  // 若调用者还没调用 setMyNumber， 则值为 `0`
  return favoriteNumber[msg.sender];
}
```

In this trivial example, anyone could call <font color="#800080"><b> setMyNumber </b></font> and store a <font color="#800080"><b> uint </b></font> in our contract, which would be tied to their address. Then when they called <font color="#800080"><b> whatIsMyNumber </b></font>, they would be returned the <font color="#800080"><b> uint </b></font> that they stored.

在这个小小的例子中，任何人都可以调用 <font color="#800080"><b> setMyNumber </b></font> 在我们的合约中存下一个 <font color="#800080"><b> uint </b></font> 并且与他们的地址相绑定。 然后，他们调用 <font color="#800080"><b> whatIsMyNumber </b></font> 就会返回他们存储的 <font color="#800080"><b> uint </b></font>。

Using <font color="#800080"><b> msg.sender </b></font> gives you the security of the Ethereum blockchain — the only way someone can modify someone else's data would be to steal the private key associated with their Ethereum address.

使用 <font color="#800080"><b> msg.sender </b></font> 很安全，因为它具有以太坊区块链的安全保障 —— 除非窃取与以太坊地址相关联的私钥，否则是没有办法修改其他人的数据的。
