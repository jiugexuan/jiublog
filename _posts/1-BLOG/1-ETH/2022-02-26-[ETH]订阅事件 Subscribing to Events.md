---
title: 【ETH】订阅事件 Subscribing to Events
date: 2022-2-26 10:00:00 +/-0800
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

As you can see, interacting with your contract via Web3.js is pretty straightforward — once you have your environment set up, calling functions and sending transactions is not all that different from a normal web API.

如你所见，通过 Web3.js 和合约交互非常简单直接——一旦你的环境建立起来， call 函数和 send 事务和普通的网络API并没有多少不同。

There's one more aspect we want to cover — subscribing to events from your contract.

还有一点东西我们想要讲到——订阅合约事件

## 监听事件 Listening for Events

If you recall from characterfactory.sol, we had an event called NewCharacter that we fired every time a new character was created:

如果你还记得 characterfactory.sol，每次新建一个人物后，我们会触发一个 NewCharacter 事件：

```solidity
event NewCharacter(uint characterId, string name, uint dna);
```

In Web3.js, you can subscribe to an event so your web3 provider triggers some logic in your code every time it fires:

在 Web3.js里， 你可以 订阅 一个事件，这样你的 Web3 提供者可以在每次事件发生后触发你的一些代码逻辑：

```solidity
cryptoCharacters.events.NewCharacter()
.on("data", function(event) {
  let character = event.returnValues;
  console.log("一个新人物诞生了！", character.characterId, character.name, character.dna);
}).on('error', console.error);
```

Note that this would trigger an alert every time ANY character was created in our DApp — not just for the current user. What if we only wanted alerts for the current user?

注意这段代码将在人物生成的时候激发一个警告信息——而不仅仅是当前用用户的人物。如果我们只想对当前用户发出提醒呢？

## 使用 indexed Using indexed

In order to filter events and only listen for changes related to the current user, our Solidity contract would have to use the <font color="#800080"><b> indexed </b></font> keyword, like we did in the <font color="#800080"><b> Transfer </b></font> event of our ERC721 implementation:

为了筛选仅和当前用户相关的事件，我们的 Solidity 合约将必须使用 <font color="#800080"><b> indexed </b></font>关键字，就像我们在 ERC721 实现中的<font color="#800080"><b> Transfer </b></font> 事件中那样：

```solidity
event Transfer(address indexed _from, address indexed _to, uint256 _tokenId);
```

In this case, because <font color="#800080"><b> _from </b></font> and <font color="#800080"><b> _to </b></font> are <font color="#800080"><b> indexed </b></font>, that means we can filter for them in our event listener in our front end:

在这种情况下， 因为<font color="#800080"><b> _from </b></font> 和<font color="#800080"><b> _to </b></font>  都是 <font color="#800080"><b> indexed </b></font>，这就意味着我们可以在前端事件监听中过滤事件

```solidity
cryptoCharacters.events.Transfer({ filter: { _to: userAccount } })
.on("data", function(event) {
  let data = event.returnValues;

  // The current user just received a Character!
  // Do something here to update the UI to show it
  // 当前用户更新了一个人物！更新界面来显示
}).on('error', console.error);
```

As you can see, using <font color="#800080"><b> events </b></font> and <font color="#800080"><b> indexed </b></font> fields can be quite a useful practice for listening to changes to your contract and reflecting them in your app's front-end.

看到了吧， 使用 <font color="#800080"><b> event </b></font> 和 <font color="#800080"><b> indexed </b></font> 字段对于监听合约中的更改并将其反映到 DApp 的前端界面中是非常有用的做法。

## 查询过去的事件 Querying past events

We can even query past events using <font color="#800080"><b> getPastEvents </b></font>, and use the filters <font color="#800080"><b> fromBlock </b></font> and <font color="#800080"><b> toBlock </b></font> to give Solidity a time range for the event logs ("block" in this case referring to the Ethereum block number):

我们甚至可以用 <font color="#800080"><b> getPastEvents </b></font> 查询过去的事件，并用过滤器 <font color="#800080"><b> fromBlock </b></font> 和 <font color="#800080"><b> toBlock </b></font> 给 Solidity 一个事件日志的时间范围("block" 在这里代表以太坊区块编号）：

```solidity
cryptoCharacters.getPastEvents("NewCharacter", { fromBlock: 0, toBlock: 'latest' })
.then(function(events) {
  // `events` is an array of `event` objects that we can iterate, like we did above
 // This code will get us a list of every character that was ever created
  // events 是可以用来遍历的 `event` 对象
  // 这段代码将返回给我们从开始以来创建的人物列表
});
```

Because you can use this method to query the event logs since the beginning of time, this presents an interesting use case: Using events as a cheaper form of storage.

因为你可以用这个方法来查询从最开始起的事件日志，这就有了一个非常有趣的用例： 用事件来作为一种更便宜的存储。

If you recall, saving data to the blockchain is one of the most expensive operations in Solidity. But using events is much much cheaper in terms of gas.

若你还能记得，在区块链上保存数据是 Solidity 中最贵的操作之一。但是用事件就便宜太多太多了。

The tradeoff here is that events are not readable from inside the smart contract itself. But it's an important use-case to keep in mind if you have some data you want to be historically recorded on the blockchain so you can read it from your app's front-end.

这里的短板是，事件不能从智能合约本身读取。但是，如果你有一些数据需要永久性地记录在区块链中以便可以在应用的前端中读取，这将是一个很好的用例。这些数据不会影响智能合约向前的状态。

For example, we could use this as a historical record of character battles — we could create an event for every time one character attacks another and who won. The smart contract doesn't need this data to calculate any future outcomes, but it's useful data for users to be able to browse from the app's front-end.

举个栗子，我们可以用事件来作为人物战斗的历史纪录——我们可以在每次人物攻击别人以及有一方胜出的时候产生一个事件。智能合约不需要这些数据来计算任何接下来的事情，但是这对我们在前端向用户展示来说是非常有用的东西。
