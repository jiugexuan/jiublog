---
title: 【ETH】调用和合约函数 Calling Contract Functions
date: 2022-2-25 11:00:00 +/-0800
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
Our contract is all set up! Now we can use Web3.js to talk to it.

我们的合约配置好了！现在来用 Web3.js 和它对话。

Web3.js has two methods we will use to call functions on our contract: <font color="#800080"><b> call </b></font> and <font color="#800080"><b> send </b></font>.

Web3.js 有两个方法来调用我们合约的函数: <font color="#800080"><b> call </b></font> and <font color="#800080"><b> send </b></font>.

## Call

<font color="#800080"><b> call </b></font> is used for <font color="#800080"><b> view </b></font> and <font color="#800080"><b> pure </b></font> functions. It only runs on the local node, and won't create a transaction on the blockchain.

<font color="#800080"><b> call </b></font> 用来调用 <font color="#800080"><b> view </b></font> 和 <font color="#800080"><b> pure </b></font> 函数。它只运行在本地节点，不会在区块链上创建事务。

> *Review: <font color="#800080"><b> view </b></font> and <font color="#800080"><b> pure </b></font> functions are read-only and don't change state on the blockchain. They also don't cost any gas, and the user won't be prompted to sign a transaction with MetaMask.<br/>复习: <font color="#800080"><b> view </b></font> 和 <font color="#800080"><b> pure </b></font> 函数是只读的并不会改变区块链的状态。它们也不会消耗任何gas。用户也不会被要求用MetaMask对事务签名。*

Using Web3.js, you would <font color="#800080"><b> call </b></font> a function named <font color="#800080"><b> myMethod </b></font> with the parameter <font color="#800080"><b> 123 </b></font> as follows:

使用 Web3.js，你可以如下 <font color="#800080"><b> call </b></font> 一个名为<font color="#800080"><b> myMethod </b></font>的方法并传入一个 <font color="#800080"><b> 123 </b></font> 作为参数：

```solidity
myContract.methods.myMethod(123).call()
```

## Send

<font color="#800080"><b> send </b></font>will create a transaction and change data on the blockchain. You'll need to use <font color="#800080"><b> send </b></font> for any functions that aren't <font color="#800080"><b> view </b></font> or <font color="#800080"><b> pure </b></font>.

<font color="#800080"><b> send </b></font> 将创建一个事务并改变区块链上的数据。你需要用 <font color="#800080"><b> send </b></font> 来调用任何非 <font color="#800080"><b> view </b></font> 或者 <font color="#800080"><b> pure </b></font> 的函数。


> *Note: <font color="#800080"><b> sending </b></font> a transaction will require the user to pay gas, and will pop up their Metamask to prompt them to sign a transaction. When we use Metamask as our web3 provider, this all happens automatically when we call <font color="#800080"><b> send() </b></font>, and we don't need to do anything special in our code. Pretty cool!<br/>注意: <font color="#800080"><b> send </b></font> 一个事务将要求用户支付gas，并会要求弹出对话框请求用户使用 Metamask 对事务签名。在我们使用 Metamask 作为我们的 web3 提供者的时候，所有这一切都会在我们调用 <font color="#800080"><b> send() </b></font> 的时候自动发生。而我们自己无需在代码中操心这一切，挺爽的吧。*

Using Web3.js, you would <font color="#800080"><b> send </b></font> a transaction calling a function named <font color="#800080"><b> myMethod </b></font> with the parameter <font color="#800080"><b> 123 </b></font> as follows:

使用 Web3.js, 你可以像这样 <font color="#800080"><b> send </b></font> 一个事务调用<font color="#800080"><b> myMethod </b></font> 并传入<font color="#800080"><b> 123 </b></font>  作为参数：

```solidity
myContract.methods.myMethod(123).send();
```

The syntax is almost identical to <font color="#800080"><b> call() </b></font>.

语法几乎 <font color="#800080"><b> call() </b></font> 一模一样。

## 获取数据 Getting Data

Now let's look at a real example of using <font color="#800080"><b> call </b></font> to access data on our contract.

来看一个使用<font color="#800080"><b> call </b></font> 读取我们合约数据的真实例子

Recall that we made our array of zombies <font color="#800080"><b> public </b></font>:

回忆一下，我们定义我们的僵尸数组为<font color="#800080"><b> 公开(public) </b></font> :

```solidity
Zombie[] public zombies;
```

In Solidity, when you declare a variable <font color="#800080"><b> public </b></font>, it automatically creates a public "getter" function with the same name. So if you wanted to look up the zombie with id <font color="#800080"><b> 15 </b></font>, you would call it as if it were a function: <font color="#800080"><b> zombies(15) </b></font>.

在 Solidity 里，当你定义一个 <font color="#800080"><b> public </b></font>变量的时候， 它将自动定义一个公开的 "getter" 同名方法， 所以如果你像要查看 id 为 <font color="#800080"><b> 15 </b></font> 的僵尸，你可以像一个函数一样调用它： <font color="#800080"><b> zombies(15) </b></font>.

Here's how we would write a JavaScript function in our front-end that would take a zombie id, query our contract for that zombie, and return the result:

这是如何在外面的前端界面中写一个 JavaScript 方法来传入一个僵尸 id，在我们的合同中查询那个僵尸并返回结果

> *Note: All the code examples we're using in this lesson are using version 1.0 of Web3.js, which uses promises instead of callbacks. Many other tutorials you'll see online are using an older version of Web3.js. The syntax changed a lot with version 1.0, so if you're copying code from other tutorials, make sure they're using the same version as you!<br/>注意: 本课中所有的示例代码都使用 Web3.js 的 1.0 版，此版本使用的是 Promises 而不是回调函数。你在线上看到的其他教程可能还在使用老版的 Web3.js。在1.0版中，语法改变了不少。如果你从其他教程中复制代码，先确保你们使用的是相同版本的Web3.js。*

```solidity
function getZombieDetails(id) {
  return cryptoZombies.methods.zombies(id).call()
}

// Call the function and do something with the result:
// 调用函数并做一些其他事情
getZombieDetails(15)
.then(function(result) {
  console.log("Zombie 15: " + JSON.stringify(result));
});
```

Let's walk through what's happening here.

我们来看看这里都做了什么

<font color="#800080"><b> cryptoZombies.methods.zombies(id).call() </b></font> will communicate with the Web3 provider node and tell it to return the zombie with index id from <font color="#800080"><b> Zombie[] public zombies </b></font> on our contract.

<font color="#800080"><b> cryptoZombies.methods.zombies(id).call() </b></font> 将和 Web3 提供者节点通信，告诉它返回从我们的合约中的 <font color="#800080"><b> Zombie[] public zombies </b></font>，id为传入参数的僵尸信息。


Note that this is asynchronous, like an API call to an external server. So Web3 returns a promise here. (If you're not familiar with JavaScript promises... Time to do some additional homework before continuing!)

注意这是 异步的，就像从外部服务器中调用API。所以 Web3 在这里返回了一个 Promises. (如果你对 JavaScript的 Promises 不了解，最好先去学习一下这方面知识再继续)。

Once the promise resolves (which means we got an answer back from the web3 provider), our example code continues with the <font color="#800080"><b> then </b></font> statement, which logs <font color="#800080"><b> result </b></font> to the console.

一旦那个 promise 被 resolve, (意味着我们从 Web3 提供者那里获得了响应)，我们的例子代码将执行 <font color="#800080"><b> then </b></font> 语句中的代码，在控制台打出 <font color="#800080"><b> result </b></font>。

<font color="#800080"><b> result </b></font> will be a javascript object that looks like this:

<font color="#800080"><b> result </b></font> 是一个像这样的 JavaScript 对象：

```solidity
{
  "name": "H4XF13LD MORRIS'S COOLER OLDER BROTHER",
  "dna": "1337133713371337",
  "level": "9999",
  "readyTime": "1522498671",
  "winCount": "999999999",
  "lossCount": "0" // Obviously.
}
```

We could then have some front-end logic to parse this object and display it in a meaningful way on the front-end.

我们可以用一些前端逻辑代码来解析这个对象并在前端界面友好展示。
