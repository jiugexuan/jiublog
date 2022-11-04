---
title: 【ETH】以太坊上的代币 Tokens on Ethereum
date: 2022-2-24 08:00:00 +/-0800
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

Let's talk about <b><font color="#0099ff">tokens</font></b>.

让我们来聊聊 <b><font color="#0099ff">代币</font></b>.

If you've been in the Ethereum space for any amount of time, you've probably heard people talking about tokens — specifically <b><font color="#0099ff">ERC20 tokens</font></b>.

如果你对以太坊的世界有一些了解，你很可能听过人们聊到代币——尤其是 <b><font color="#0099ff">ERC20 代币</font></b>.

A <b><font color="#0099ff">tokens</font></b> on Ethereum is basically just a smart contract that follows some common rules — namely it implements a standard set of functions that all other token contracts share, such as font color="#800080"><b> transfer(address _to, uint256 _value) </b></font>  and <font color="#800080"><b> balanceOf(address _owner) </b></font>.

一个 <b><font color="#0099ff">代币</font></b> 在以太坊基本上就是一个遵循一些共同规则的智能合约——即它实现了所有其他代币合约共享的一组标准函数，例如<font color="#800080"><b> transfer(address _to, uint256 _value) </b></font>  和<font color="#800080"><b> balanceOf(address _owner) </b></font> .

Internally the smart contract usually has a mapping, mapping(address => uint256) balances, that keeps track of how much balance each address has.

在智能合约内部，通常有一个映射， mapping(address => uint256) balances，用于追踪每个地址还有多少余额。

So basically a token is just a contract that keeps track of who owns how much of that token, and some functions so those users can transfer their tokens to other addresses.

所以基本上一个代币只是一个追踪谁拥有多少该代币的合约，和一些可以让那些用户将他们的代币转移到其他地址的函数。

## 它为什么重要呢？ Why does it matter?

Since all ERC20 tokens share the same set of functions with the same names, they can all be interacted with in the same ways.

由于所有 ERC20 代币共享具有相同名称的同一组函数，它们都可以以相同的方式进行交互。

This means if you build an application that is capable of interacting with one ERC20 token, it's also capable of interacting with any ERC20 token. That way more tokens can easily be added to your app in the future without needing to be custom coded. You could simply plug in the new token contract address, and boom, your app has another token it can use.

这意味着如果你构建的应用程序能够与一个 ERC20 代币进行交互，那么它就也能够与任何 ERC20 代币进行交互。 这样一来，将来你就可以轻松地将更多的代币添加到你的应用中，而无需进行自定义编码。 你可以简单地插入新的代币合约地址，然后哗啦，你的应用程序有另一个它可以使用的代币了。

One example of this would be an exchange. When an exchange adds a new ERC20 token, really it just needs to add another smart contract it talks to. Users can tell that contract to send tokens to the exchange's wallet address, and the exchange can tell the contract to send the tokens back out to users when they request a withdraw.

其中一个例子就是交易所。 当交易所添加一个新的 ERC20 代币时，实际上它只需要添加与之对话的另一个智能合约。 用户可以让那个合约将代币发送到交易所的钱包地址，然后交易所可以让合约在用户要求取款时将代币发送回给他们。

The exchange only needs to implement this transfer logic once, then when it wants to add a new ERC20 token, it's simply a matter of adding the new contract address to its database.

交易所只需要实现这种转移逻辑一次，然后当它想要添加一个新的 ERC20 代币时，只需将新的合约地址添加到它的数据库即可。

## 其他代币标准 Other token standards

ERC20 tokens are really cool for tokens that act like currencies. But it's not particularly useful for artworks.

对于像货币一样的代币来说，ERC20 代币非常酷。 但是对于艺术品来说就并不是特别有用。

For one,  artworks aren't divisible like currencies — I can send you 0.237 ETH, but transfering you 0.237 of a artworks doesn't really make sense.Secondly, all artworks are not created equal.

首先，艺术品不像货币可以分割 —— 我可以发给你 0.237 以太，但是转移给你 0.237 的艺术品听起来就有些搞笑。其次，并不是所有艺术品是平等的。

There's another token standard that's a much better fit for crypto-collectibles  — and they're called <b><font color="#0099ff"> ERC721 tokens</font></b>.

有另一个代币标准更适合如这样的加密收藏品——它们被称为<b><font color="#0099ff"> ERC721 代币</font></b>。

<b><font color="#0099ff"> ERC721 tokens</font></b> are not interchangeable since each one is assumed to be unique, and are not divisible. You can only trade them in whole units, and each one has a unique ID.

<b><font color="#0099ff"> ERC721 代币 </font></b>是不能互换的，因为每个代币都被认为是唯一且不可分割的。 你只能以整个单位交易它们，并且每个单位都有唯一的 ID。

> *Note that using a standard like ERC721 has the benefit that we don't have to implement the auction or escrow logic within our contract that determines how players can trade / sell our assets. If we conform to the spec, someone else could build an exchange platform for crypto-tradable ERC721 assets, and our ERC721 zombies would be usable on that platform. So there are clear benefits to using a token standard instead of rolling your own trading logic.<br/>请注意，使用像 ERC721 这样的标准的优势就是，我们不必在我们的合约中实现拍卖或托管逻辑，这决定了玩家能够如何交易／出售我们的资产。 如果我们符合规范，其他人可以为加密可交易的 ERC721 资产搭建一个交易所平台，我们的 ERC721 僵尸将可以在该平台上使用。 所以使用代币标准相较于使用你自己的交易逻辑有明显的好处。*
