---
title: 【ETH】可支付 Payable
date: 2022-2-23 08:00:00 +/-0800
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

## 可支付 Payable

Up until now, we've covered quite a few function <b><font color="#0099ff"> modifiers </font></b>. It can be difficult to try to remember everything, so let's run through a quick review:

截至目前，我们只接触到很少的<b><font color="#0099ff"> 函数修饰符 </font></b>。 要记住所有的东西很难，所以我们来个概览：

  1. We have visibility modifiers that control when and where the function can be called from: <font color="#800080"><b> private </b></font> means it's only callable from other functions inside the contract; <font color="#800080"><b> internal </b></font> is like <font color="#800080"><b> private </b></font> but can also be called by contracts that inherit from this one; <font color="#800080"><b> external </b></font> can only be called outside the contract; and finally <font color="#800080"><b> public </b></font> can be called anywhere, both internally and externally.</br>我们有决定函数何时和被谁调用的可见性修饰符: <font color="#800080"><b> private </b></font> 意味着它只能被合约内部调用； <font color="#800080"><b> internal </b></font> 就像 <font color="#800080"><b> private </b></font> 但是也能被继承的合约调用； <font color="#800080"><b> external </b></font> 只能从合约外部调用；最后 <font color="#800080"><b> public </b></font> 可以在任何地方调用，不管是内部还是外部。<br/>
  2. We also have state modifiers, which tell us how the function interacts with the BlockChain: <font color="#800080"><b> view </b></font> tells us that by running the function, no data will be saved/changed. <font color="#800080"><b> pure </b></font> tells us that not only does the function not save any data to the blockchain, but it also doesn't read any data from the blockchain. Both of these don't cost any gas to call if they're called externally from outside the contract (but they do cost gas if called internally by another function).<br/>我们也有状态修饰符， 告诉我们函数如何和区块链交互: <font color="#800080"><b> view </b></font> 告诉我们运行这个函数不会更改和保存任何数据； <font color="#800080"><b> pure </b></font> 告诉我们这个函数不但不会往区块链写数据，它甚至不从区块链读取数据。这两种在被从合约外部调用的时候都不花费任何gas（但是它们在被内部其他函数调用的时候将会耗费gas）。<br/>
  3. Then we have custom <font color="#800080"><b> modifiers </b></font> : <font color="#800080"><b> onlyOwner </b></font> and <font color="#800080"><b> aboveLevel </b></font>, for example. For these we can define custom logic to determine how they affect a function.<br/>然后我们有了自定义的 <font color="#800080"><b> modifiers </b></font> : <font color="#800080"><b> onlyOwner </b></font> 和 <font color="#800080"><b> aboveLevel </b></font>。 对于这些修饰符我们可以自定义其对函数的约束逻辑。

These modifiers can all be stacked together on a function definition as follows:

这些修饰符可以同时作用于一个函数定义上：

```solidity
function test() external view onlyOwner anotherModifier { /* ... */ }
```

In this chapter, we're going to introduce one more function modifier: <font color="#800080"><b> payable </b></font>.

在这一章，我们来学习一个新的修饰符<font color="#800080"><b> payable </b></font>。

## payable 修饰符 payable modifier

<font color="#800080"><b> payable </b></font> functions are part of what makes Solidity and Ethereum so cool — they are a special type of function that can receive Ether.

<font color="#800080"><b> payable </b></font> 方法是让 Solidity 和以太坊变得如此酷的一部分 —— 它们是一种可以接收以太的特殊函数。

Let that sink in for a minute. When you call an API function on a normal web server, you can't send US dollars along with your function call — nor can you send Bitcoin.

先放一下。当你在调用一个普通网站服务器上的API函数的时候，你无法用你的函数传送美元——你也不能传送比特币。

But in Ethereum, because both the money (Ether), the data (transaction payload), and the contract code itself all live on Ethereum, it's possible for you to call a function and pay money to the contract at the same time.

但是在以太坊中， 因为钱 (以太), 数据 (事务负载)， 以及合约代码本身都存在于以太坊。你可以在同时调用函数 并付钱给另外一个合约。

This allows for some really interesting logic, like requiring a certain payment to the contract in order to execute a function.

这就允许出现很多有趣的逻辑， 比如向一个合约要求支付一定的钱来运行一个函数。

Let's look at an example

### 来看个例子

```solidity
contract OnlineStore {
  function buySomething() external payable {
    // Check to make sure 0.001 ether was sent to the function call:
    // 检查以确定0.001以太发送出去来运行函数:
    require(msg.value == 0.001 ether);
    // If so, some logic to transfer the digital item to the caller of the function:
    // 如果为真，一些用来向函数调用者发送数字内容的逻辑
    transferThing(msg.sender);
  }
}
```

Here, <font color="#800080"><b> msg.value </b></font> is a way to see how much Ether was sent to the contract, and <font color="#800080"><b> ether </b></font> is a built-in unit.

在这里，<font color="#800080"><b> msg.value </b></font> 是一种可以查看向合约发送了多少以太的方法，另外 <font color="#800080"><b> ether </b></font> 是一个內建单元。

What happens here is that someone would call the function from web3.js (from the DApp's JavaScript front-end) as follows:

这里发生的事是，一些人会从 web3.js 调用这个函数 (从DApp的前端)， 像这样 :

```solidity
// Assuming `OnlineStore` points to your contract on Ethereum:
// 假设 `OnlineStore` 在以太坊上指向你的合约:
OnlineStore.buySomething().send(from: web3.eth.defaultAccount, value: web3.utils.toWei(0.001))
```

Notice the <font color="#800080"><b> value </b></font> field, where the javascript function call specifies how much <font color="#800080"><b> ether </b></font> to send (0.001). If you think of the transaction like an envelope, and the parameters you send to the function call are the contents of the letter you put inside, then adding a <font color="#800080"><b> value </b></font> is like putting cash inside the envelope — the letter and the money get delivered together to the recipient.

注意这个 <font color="#800080"><b> value </b></font> 字段， JavaScript 调用来指定发送多少(0.001)<font color="#800080"><b> 以太 </b></font>。如果把事务想象成一个信封，你发送到函数的参数就是信的内容。 添加一个 <font color="#800080"><b> value </b></font> 很像在信封里面放钱 —— 信件内容和钱同时发送给了接收者。

> *Note: If a function is not marked <font color="#800080"><b> payable </b></font> and you try to send Ether to it as above, the function will reject your transaction.<br/>注意： 如果一个函数没标记为<font color="#800080"><b> payable</b></font>， 而你尝试利用上面的方法发送以太，函数将拒绝你的事务。*
