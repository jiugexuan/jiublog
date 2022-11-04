---
title: 【ETH】MetaMask 和账户 Metamask & Accounts
date: 2022-2-25 12:00:00 +/-0800
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

Awesome! You've successfully written front-end code to interact with your first smart contract.

太棒了！你成功地写了一些前端代码来和你的第一个智能合约交互。

But our Solidity contract is expecting <font color="#800080"><b> owner </b></font> to be a Solidity address. How can we know the <font color="#800080"><b> address </b></font> of the user using our app?

我们的 Solidity 合约需要 <font color="#800080"><b> owner </b></font> 作为 Solidity <font color="#800080"><b> address </b></font>。我们如何能知道应用用户的地址呢？

## 获得 MetaMask中的用户账户 Getting the user's account in MetaMask

MetaMask allows the user to manage multiple accounts in their extension.

MetaMask 允许用户在扩展中管理多个账户。

We can see which account is currently active on the injected <font color="#800080"><b> web3 </b></font> variable via:

我们可以通过这样来获取 <font color="#800080"><b> web3 </b></font> 变量中激活的当前账户：

```solidity
var userAccount = web3.eth.accounts[0];
```

Because the user can switch the active account at any time in MetaMask, our app needs to monitor this variable to see if it has changed and update the UI accordingly. For example, if the user's homepage displays their assets, when they change their account in MetaMask, we'll want to update the page to show the assets for the new account they've selected.

因为用户可以随时在 MetaMask 中切换账户，我们的应用需要监控这个变量，一旦改变就要相应更新界面。例如，若用户的首页展示它们的资产，当他们在 MetaMask 中切换了账号，我们就需要更新页面来展示新选择的账户的资产。

We can do that with a <font color="#800080"><b> setInterval </b></font> loop as follows:

我们可以通过 <font color="#800080"><b> setInterval </b></font> 方法来做:

```solidity
var accountInterval = setInterval(function() {
  // Check if account has changed
  // 检查账户是否切换
  if (web3.eth.accounts[0] !== userAccount) {
    userAccount = web3.eth.accounts[0];
    // Call some function to update the UI with the new account
    // 调用一些方法来更新界面
    updateInterface();
  }
}, 100);
```
What this does is check every 100 milliseconds to see if <font color="#800080"><b> userAccount </b></font> is still equal <font color="#800080"><b> web3.eth.accounts[0] </b></font>(i.e. does the user still have that account active). If not, it reassigns <font color="#800080"><b> userAccount </b></font> to the currently active account, and calls a function to update the display.

这段代码做的是，每100毫秒检查一次 <font color="#800080"><b> userAccount </b></font> 是否还等于 <font color="#800080"><b> web3.eth.accounts[0] </b></font> (比如：用户是否还激活了那个账户)。若不等，则将 当前激活用户赋值给 <font color="#800080"><b> userAccount </b></font>，然后调用一个函数来更新界面。
