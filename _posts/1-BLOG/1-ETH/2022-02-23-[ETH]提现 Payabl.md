---
title: 【ETH】提现 Payable
date: 2022-2-23 10:00:00 +/-0800
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

Now that we know how to send Ether to the contract, we obviously want to know what happens after we send it?

在了解如何向合约发送以太后，我们会显然想知道那么在发送之后会发生什么呢？

After you send Ether to a contract, it gets stored in the contract's Ethereum account, and it will be trapped there — unless you add a function to withdraw the Ether from the contract.

在你发送以太之后，它将被存储进以合约的以太坊账户中， 并冻结在哪里 —— 除非你添加一个函数来从合约中把以太提现。

You can write a function to withdraw Ether from the contract as follows:

你可以写一个函数来从合约中提现以太，类似这样：

```solidity
contract GetPaid is Ownable {
  function withdraw() external onlyOwner {
    owner.transfer(this.balance);
  }
}
```

Note that we're using <font color="#800080"><b> owner() </b></font> and <font color="#800080"><b> onlyOwner </b></font> from the <font color="#800080"><b> Ownable </b></font> contract, assuming that was imported.

注意我们使用 <font color="#800080"><b> Ownable </b></font> 合约中的 <font color="#800080"><b> owner() </b></font> 和 <font color="#800080"><b> Ownable </b></font>，假定它已经被引入了。

It is important to note that you cannot transfer Ether to an address unless that address is of type <font color="#800080"><b> address payable </b></font>. But the <font color="#800080"><b> _owner </b></font> variable is of type <font color="#800080"><b> uint160 </b></font>, meaning that we must explicitly cast it to <font color="#800080"><b> address payable </b></font>.

需要注意的是，您不能将 Ether 转移到某个地址，除非该地址属于<font color="#800080"><b> 应付地址 </b></font>类型。 但是 <font color="#800080"><b> _owner </b></font> 变量是 <font color="#800080"><b> uint160 </b></font> 类型的，这意味着我们必须将其显式转换为<font color="#800080"><b> 应付地址 </b></font>。

Once you cast the address from <font color="#800080"><b> uint160 </b></font> to <font color="#800080"><b> address payable </b></font>, you can transfer Ether to that address using the <font color="#800080"><b> transfer </b></font> function, and <font color="#800080"><b> address(this).balance </b></font> will return the total balance stored on the contract. So if 100 users had paid 1 Ether to our contract, <font color="#800080"><b> address(this).balance </b></font> would equal 100 Ether.

将地址从 <font color="#800080"><b> uint160 </b></font> 转换为 <font color="#800080"><b> 应付地址 </b></font>后，您可以使用 <font color="#800080"><b> transfer </b></font> 函数将 Ether 转移到该地址， 然后 <font color="#800080"><b> this.balance </b></font> 将返回当前合约存储了多少以太。 所以如果100个用户每人向我们支付1以太， <font color="#800080"><b> this.balance </b></font> 将是100以太。

You can use <font color="#800080"><b> transfer </b></font> to send funds to any Ethereum address. For example, you could have a function that transfers Ether back to the <font color="#800080"><b> msg.sender </b></font> if they overpaid for an item:

你可以通过 <font color="#800080"><b> transfer </b></font> 向任何以太坊地址付钱。 比如，你可以有一个函数在 <font color="#800080"><b> msg.sender </b></font> 超额付款的时候给他们退钱：

```solidity
uint itemFee = 0.001 ether;
msg.sender.transfer(msg.value - itemFee);
```

Or in a contract with a buyer and a seller, you could save the seller's address in storage, then when someone purchases his item, transfer him the fee paid by the buyer: <font color="#800080"><b> seller.transfer(msg.value) </b></font>.

或者在一个有卖家和卖家的合约中， 你可以把卖家的地址存储起来， 当有人买了它的东西的时候，把买家支付的钱发送给它 <font color="#800080"><b> seller.transfer(msg.value) </b></font>。

These are some examples of what makes Ethereum programming really cool — you can have decentralized marketplaces like this that aren't controlled by anyone.

有很多例子来展示什么让以太坊编程如此之酷 —— 你可以拥有一个不被任何人控制的去中心化市场。
