---
title: 【ETH】Web3 提供者 Web3 Providers
date: 2022-2-25 09:00:00 +/-0800
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

Great! Now that we have Web3.js in our project, let's get it initialized and talking to the blockchain.

太棒了。现在我们的项目中有了Web3.js, 来初始化它然后和区块链对话吧。

The first thing we need is a <b><font color="#0099ff">Web3 Provider</font></b>.

首先我们需要  <b><font color="#0099ff">Web3 Provider</font></b>.

Remember, Ethereum is made up of <b><font color="#0099ff"> nodes </font></b> that all share a copy of the same data. Setting a Web3 Provider in Web3.js tells our code which node we should be talking to handle our reads and writes. It's kind of like setting the URL of the remote web server for your API calls in a traditional web app.

要记住，以太坊是由共享同一份数据的相同拷贝的 <b><font color="#0099ff"> 节点 </font></b> 构成的。 在 Web3.js 里设置 Web3 的 Provider（提供者） 告诉我们的代码应该和 哪个节点 交互来处理我们的读写。这就好像在传统的 Web 应用程序中为你的 API 调用设置远程 Web 服务器的网址。

You could host your own Ethereum node as a provider. However, there's a third-party service that makes your life easier so you don't need to maintain your own Ethereum node in order to provide a DApp for your users — <b><font color="#0099ff"> Infura </font></b>.

你可以运行你自己的以太坊节点来作为 Provider。 不过，有一个第三方的服务，可以让你的生活变得轻松点，让你不必为了给你的用户提供DApp而维护一个以太坊节点— <b><font color="#0099ff"> Infura </font></b>.

## Infura

 <b><font color="#0099ff"> Infura </font></b> is a service that maintains a set of Ethereum nodes with a caching layer for fast reads, which you can access for free through their API. Using Infura as a provider, you can reliably send and receive messages to/from the Ethereum blockchain without needing to set up and maintain your own node.

<b><font color="#0099ff"> Infura </font></b>是一个服务，它维护了很多以太坊节点并提供了一个缓存层来实现高速读取。你可以用他们的 API 来免费访问这个服务。 用 Infura 作为节点提供者，你可以不用自己运营节点就能很可靠地向以太坊发送、接收信息。

You can set up Web3 to use Infura as your web3 provider as follows:

你可以通过这样把 Infura 作为你的 Web3 节点提供者：

```solidity
var web3 = new Web3(new Web3.providers.WebsocketProvider("wss://mainnet.infura.io/ws"));
```

However, since our DApp is going to be used by many users — and these users are going to WRITE to the blockchain and not just read from it — we'll need a way for these users to sign transactions with their private key.

不过，因为我们的 DApp 将被很多人使用，这些用户不单会从区块链读取信息，还会向区块链 _写_ 入信息，我们需要用一个方法让用户可以用他们的私钥给事务签名。

> *Note: Ethereum (and blockchains in general) use a public / private key pair to digitally sign transactions. Think of it like an extremely secure password for a digital signature. That way if I change some data on the blockchain, I can prove via my public key that I was the one who signed it — but since no one knows my private key, no one can forge a transaction for me.<br/>注意: 以太坊 (以及通常意义上的 blockchains )使用一个公钥/私钥对来对给事务做数字签名。把它想成一个数字签名的异常安全的密码。这样当我修改区块链上的数据的时候，我可以用我的公钥来 证明 我就是签名的那个。但是因为没人知道我的私钥，所以没人能伪造我的事务。*

Cryptography is complicated, so unless you're a security expert and you really know what you're doing, it's probably not a good idea to try to manage users' private keys yourself in our app's front-end.

加密学非常复杂，所以除非你是个专家并且的确知道自己在做什么，你最好不要在你应用的前端中管理你用户的私钥。

But luckily you don't need to — there are already services that handle this for you. The most popular of these is Metamask.

不过幸运的是，你并不需要，已经有可以帮你处理这件事的服务了： Metamask.

## Metamask

<b><font color="#0099ff">Metamask</font></b> is a browser extension for Chrome and Firefox that lets users securely manage their Ethereum accounts and private keys, and use these accounts to interact with websites that are using Web3.js. (If you haven't used it before, you'll definitely want to go and install it — then your browser is Web3 enabled, and you can now interact with any website that communicates with the Ethereum blockchain!).

<b><font color="#0099ff">Metamask</font></b> 是 Chrome 和 Firefox 的浏览器扩展， 它能让用户安全地维护他们的以太坊账户和私钥， 并用他们的账户和使用 Web3.js 的网站互动（如果你还没用过它，你肯定会想去安装的——这样你的浏览器就能使用 Web3.js 了，然后你就可以和任何与以太坊区块链通信的网站交互了）

And as a developer, if you want users to interact with your DApp through a website in their web browser (like we're doing with our CryptoZombies game), you'll definitely want to make it Metamask-compatible.

作为开发者，如果你想让用户从他们的浏览器里通过网站和你的DApp交互（就像我们在 CryptoZombies 游戏里一样），你肯定会想要兼容 Metamask 的。

> *Note: Metamask uses Infura's servers under the hood as a web3 provider, just like we did above — but it also gives the user the option to choose their own web3 provider. So by using Metamask's web3 provider, you're giving the user a choice, and it's one less thing you have to worry about in your app.<br/>注意: Metamask 默认使用 Infura 的服务器做为 web3 提供者。 就像我们上面做的那样。不过它还为用户提供了选择他们自己 Web3 提供者的选项。所以使用 Metamask 的 web3 提供者，你就给了用户选择权，而自己无需操心这一块。*

## 使用 Metamask 的 web3 提供者 Using Metamask's web3 provider

Metamask injects their web3 provider into the browser in the global JavaScript object <font color="#800080"><b> web3 </b></font>. So your app can check to see if <font color="#800080"><b> web3 </b></font> exists, and if it does use <font color="#800080"><b> web3.currentProvider </b></font> as its provider.

Metamask 把它的 <font color="#800080"><b> web3 </b></font> 提供者注入到浏览器的全局 JavaScript对象web3中。所以你的应用可以检查 <font color="#800080"><b> web3 </b></font> 是否存在。若存在就使用 <font color="#800080"><b> web3.currentProvider </b></font> 作为它的提供者。

Here's some template code provided by Metamask for how we can detect to see if the user has Metamask installed, and if not tell them they'll need to install it to use our app:

这里是一些 Metamask 提供的示例代码，用来检查用户是否安装了MetaMask，如果没有安装就告诉用户需要安装MetaMask来使用我们的应用。

```solidity
window.addEventListener('load', function() {
  // Checking if Web3 has been injected by the browser (Mist/MetaMask)
  // 检查web3是否已经注入到(Mist/MetaMask)
  if (typeof web3 !== 'undefined') {
    // Use Mist/MetaMask's provider
    // 使用 Mist/MetaMask 的提供者
    web3js = new Web3(web3.currentProvider);
  } else {
    // Handle the case where the user doesn't have web3. Probably
    // show them a message telling them to install Metamask in
    // order to use our app.
    // 处理用户没安装的情况， 比如显示一个消息
    // 告诉他们要安装 MetaMask 来使用我们的应用
  }

  // Now you can start your app & access web3js freely:
  // 现在你可以启动你的应用并自由访问 Web3.js:
  startApp()

})
```

You can use this boilerplate code in all the apps you create in order to require users to have Metamask to use your DApp.

你可以在你所有的应用中使用这段样板代码，好检查用户是否安装以及告诉用户安装 MetaMask。

> *Note: There are other private key management programs your users might be using besides MetaMask, such as the web browser Mist. However, they all implement a common pattern of injecting the variable <font color="#800080"><b> web3 </b></font>, so the method we describe here for detecting the user's web3 provider will work for these as well.<br/>注意: 除了MetaMask，你的用户也可能在使用其他他的私钥管理应用，比如 Mist 浏览器。不过，它们都实现了相同的模式来注入 <font color="#800080"><b> web3 </b></font> 变量。所以我这里描述的方法对两者是通用的。*
