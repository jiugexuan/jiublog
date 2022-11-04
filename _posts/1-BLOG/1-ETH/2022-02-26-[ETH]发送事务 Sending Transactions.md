---
title: 【ETH】发送事务 Calling Payable Functions
date: 2022-2-26 09:00:00 +/-0800
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

Awesome! Now our UI will detect the user's metamask account, and automatically display their assets on the homepage.

这下我们的界面能检测用户的 MetaMask 账户，并自动在首页显示它们的资产了，有没有很棒？

Now let's look at using <font color="#800080"><b> send </b></font> functions to change data on our smart contract.

现在我们来看看用 <font color="#800080"><b> send </b></font> 函数来修改我们智能合约里面的数据。

There are a few major differences from <font color="#800080"><b> call </b></font> functions:

相对 <font color="#800080"><b> call </b></font> 函数，send 函数有如下主要区别:

  1. <font color="#800080"><b> sending </b></font> a transaction requires a from address of who's calling the function (which becomes msg.sender in your Solidity code). We'll want this to be the user of our DApp, so MetaMask will pop up to prompt them to sign the transaction.<br/><font color="#800080"><b> send </b></font> 一个事务需要一个 from 地址来表明谁在调用这个函数（也就是你 Solidity 代码里的 msg.sender )。 我们需要这是我们 DApp 的用户，这样一来 MetaMask 才会弹出提示让他们对事务签名。

  2. font color="#800080"><b> send </b></font>sending a transaction costs gas<br/>font color="#800080"><b> send </b></font> 一个事务将花费 gas

  3. There will be a significant delay from when the user <font color="#800080"><b> sends </b></font> a transaction and when that transaction actually takes effect on the blockchain. This is because we have to wait for the transaction to be included in a block, and the block time for Ethereum is on average 15 seconds. If there are a lot of pending transactions on Ethereum or if the user sends too low of a gas price, our transaction may have to wait several blocks to get included, and this could take minutes.<br/>在用户 <font color="#800080"><b> send </b></font> 一个事务到该事务对区块链产生实际影响之间有一个不可忽略的延迟。这是因为我们必须等待事务被包含进一个区块里，以太坊上一个区块的时间平均下来是15秒左右。如果当前在以太坊上有大量挂起事务或者用户发送了过低的 gas 价格，我们的事务可能需要等待数个区块才能被包含进去，往往可能花费数分钟。

Thus we'll need logic in our app to handle the asynchronous nature of this code.

所以在我们的代码中我们需要编写逻辑来处理这部分异步特性。

## 例子 Example

Let's look at an example with the first function in our contract a new user will call: <font color="#800080"><b> createRandomCharacter </b></font>.

我们来看一个合约中一个新用户将要调用的第一个函数: <font color="#800080"><b> createRandomCharacter </b></font>.


```solidity
function createRandomCharacter(string _name) public {
  require(ownerCharacterCount[msg.sender] == 0);
  uint randDna = _generateRandomDna(_name);
  randDna = randDna - randDna % 100;
  _createCharacter(_name, randDna);
}
```

Here's an example of how we could call this function in Web3.js using MetaMask:

这是如何在用 MetaMask 在 Web3.js 中调用这个函数的示例:

```solidity
function createRandomCharacter(name) {

  // This is going to take a while, so update the UI to let the user know
  // the transaction has been sent
  // 这将需要一段时间，所以在界面中告诉用户这一点
  // 事务被发送出去了
  $("#txStatus").text("正在区块链上创建人物，这将需要一会儿...");
  // Send the tx to our contract:
  // 把事务发送到我们的合约:
  return cryptoCharacters.methods.createRandomCharacter(name)
  .send({ from: userAccount })
  .on("receipt", function(receipt) {
    $("#txStatus").text("成功生成了 " + name + "!");
    // Transaction was accepted into the blockchain, let's redraw the UI
    // 事务被区块链接受了，重新渲染界面
    getCharactersByOwner(userAccount).then(displayZombies);
  })
  .on("error", function(error) {
     // Do something to alert the user their transaction has failed
    // 告诉用户合约失败了
    $("#txStatus").text(error);
  });
}
```
Our function <font color="#800080"><b> sends </b></font> a transaction to our Web3 provider, and chains some event listeners:

我们的函数 <font color="#800080"><b> send </b></font> 一个事务到我们的 Web3 提供者，然后链式添加一些事件监听:

 - <font color="#800080"><b> receipt </b></font> will fire when the transaction is included into a block on Ethereum, which means our zombie has been created and saved on our contract.<br/><font color="#800080"><b> receipt </b></font> 将在合约被包含进以太坊区块上以后被触发，这意味着僵尸被创建并保存进我们的合约了。
 - <font color="#800080"><b> error </b></font> will fire if there's an issue that prevented the transaction from being included in a block, such as the user not sending enough gas. We'll want to inform the user in our UI that the transaction didn't go through so they can try again.<br/> <font color="#800080"><b> error </b></font> 将在事务未被成功包含进区块后触发，比如用户未支付足够的 gas。我们需要在界面中通知用户事务失败以便他们可以再次尝试。
>* Note: You can optionally specify  <font color="#800080"><b> gas </b></font> and <font color="#800080"><b> gasPrice </b></font> when you call <font color="#800080"><b> send </b></font>, e.g. <font color="#800080"><b> .send({ from: userAccount, gas: 3000000 }) </b></font>. If you don't specify this, MetaMask will let the user choose these values.<br/>注意:你可以在调用 <font color="#800080"><b> send </b></font> 时选择指定 <font color="#800080"><b> gas </b></font> 和 <font color="#800080"><b> gasPrice </b></font>， 例如： <font color="#800080"><b> .send({ from: userAccount, gas: 3000000 }) </b></font>。如果你不指定，MetaMask 将让用户自己选择数值。*
