---
title: 【ETH】利用 'View' 函数节省 Gas Saving Gas With 'View' Functions
date: 2022-2-22 13:00:00 +/-0800
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

## “view” 函数不花 “gas” View functions don't cost gas

<font color="#800080"><b>view </b></font> functions don't cost any gas when they're called externally by a user.

当玩家从外部调用一个<font color="#800080"><b> view </b></font>函数，是不需要支付一分 gas 的。

This is because <font color="#800080"><b> view </b></font> functions don't actually change anything on the blockchain – they only read the data. So marking a function with <font color="#800080"><b> view </b></font> tells web3.js that it only needs to query your local Ethereum node to run the function, and it doesn't actually have to create a transaction on the blockchain (which would need to be run on every single node, and cost gas).

这是因为 <font color="#800080"><b> view </b></font> 函数不会真正改变区块链上的任何数据 - 它们只是读取。因此用 <font color="#800080"><b> view </b></font> 标记一个函数，意味着告诉 web3.js，运行这个函数只需要查询你的本地以太坊节点，而不需要在区块链上创建一个事务（事务需要运行在每个节点上，因此花费 gas）。

We'll cover setting up <font color="#800080"><b> web3.js </b></font> with your own node later. But for now the big takeaway is that you can optimize your DApp's gas usage for your users by using read-only <font color="#800080"><b> external view </b></font> functions wherever possible.

稍后我们将介绍如何在自己的节点上设置 <font color="#800080"><b> web3.js </b></font>。但现在，你关键是要记住，在所能只读的函数上标记上表示“只读”的 <font color="#800080"><b> external view </b></font> 声明，就能为你的玩家减少在 DApp 中 gas 用量。

> *Note: If a <font color="#800080"><b> view </b></font> function is called internally from another function in the same contract that is not a <font color="#800080"><b> view </b></font> function, it will still cost gas. This is because the other function creates a transaction on Ethereum, and will still need to be verified from every node. So <font color="#800080"><b> view </b></font> functions are only free when they're called externally.<br/>注意：如果一个 <font color="#800080"><b> view </b></font> 函数在另一个函数的内部被调用，而调用函数与 <font color="#800080"><b> view </b></font> 函数的不属于同一个合约，也会产生调用成本。这是因为如果主调函数在以太坊创建了一个事务，它仍然需要逐个节点去验证。所以标记为 <font color="#800080"><b> view </b></font> 的函数只有在外部调用时才是免费的。*
