---
title: 【ETH】ERC721 标准 ERC721 standard
date: 2022-2-24 09:00:00 +/-0800
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

Let's take a look at the ERC721 standard:

让我们来看一看 ERC721 标准：

```solidity
contract ERC721 {
  event Transfer(address indexed _from, address indexed _to, uint256 _tokenId);
  event Approval(address indexed _owner, address indexed _approved, uint256 _tokenId);

  function balanceOf(address _owner) public view returns (uint256 _balance);
  function ownerOf(uint256 _tokenId) public view returns (address _owner);
  function transfer(address _to, uint256 _tokenId) public;
  function approve(address _to, uint256 _tokenId) public;
  function takeOwnership(uint256 _tokenId) public;
}
```

This is the list of methods we'll need to implement, which we'll be doing in pieces.

这是我们需要实现的方法列表，我们将在接下来的逐个学习。

## 实现一个代币合约 Implement a token contract

When implementing a token contract, the first thing we do is copy the interface to its own Solidity file and import it, <font color="#800080"><b> import "./erc721.sol" </b></font>;. Then we have our contract inherit from it, and we override each method with a function definition.

在实现一个代币合约的时候，我们首先要做的是将接口复制到它自己的 Solidity 文件并导入它，<font color="#800080"><b> import "./erc721.sol" </b></font>;。 接着，让我们的合约继承它，然后我们用一个函数定义来重写每个方法。

In Solidity, your contract can inherit from multiple contracts as follows:

在Solidity，你的合约可以继承自多个合约，参考如下：

```solidity

contract SatoshiNakamoto is NickSzabo, HalFinney {
  // Omg, the secrets of the universe revealed!
  // 啧啧啧，宇宙的奥秘泄露了
}

```

As you can see, when using multiple inheritance, you just separate the multiple contracts you're inheriting from with a comma, ,. In this case, our contract is inheriting from <font color="#800080"><b> NickSzabo </b></font> and <font color="#800080"><b> HalFinney </b></font>.

正如你所见，当使用多重继承的时候，你只需要用逗号 , 来隔开几个你想要继承的合约。在上面的例子中，我们的合约继承自<font color="#800080"><b> NickSzabo </b></font>  和 <font color="#800080"><b> HalFinney </b></font>。

## balanceOf & ownerOf

Great, let's dive into the ERC721 implementation!

太棒了，我们来深入讨论一下 ERC721 的实现。

We've gone ahead and copied the empty shell of all the functions you'll be implementing in this lesson

我们已经把所有你需要在本课中实现的函数的空壳复制好了。

In this chapter, we're going to implement the first two methods: <font color="#800080"><b> balanceOf </b></font> and <font color="#800080"><b> ownerOf </b></font>.

在本章节，我们将实现头两个方法： <font color="#800080"><b> balanceOf </b></font> 和 <font color="#800080"><b> ownerOf </b></font>。

### balanceOf

```solidity
  function balanceOf(address _owner) public view returns (uint256 _balance);
```

This function simply takes an <font color="#800080"><b>  address </b></font>, and returns how many tokens that <font color="#800080"><b>  address </b></font> owns.

这个函数只需要一个传入 <font color="#800080"><b>  address </b></font>参数，然后返回这个 <font color="#800080"><b>  address </b></font> 拥有多少代币。


### ownerOf

```solidity
  function ownerOf(uint256 _tokenId) public view returns (address _owner);
```

This function takes a token ID , and returns the <font color="#800080"><b>  address </b></font> of the person who owns it.

这个函数需要传入一个代币 ID 作为参数 ，然后返回该代币拥有者的 <font color="#800080"><b>  address </b></font>。

Again, this is very straightforward for us to implement, since we already have a <font color="#800080"><b>  mapping (映射) </b></font> in our DApp that stores this information. We can implement this function in one line, just a <font color="#800080"><b>  return </b></font> statement.

同样的，因为在我们的 DApp 里已经有一个 <font color="#800080"><b>  mapping (映射) </b></font> 存储了这个信息，所以对我们来说这个实现非常直接清晰。我们可以只用一行 <font color="#800080"><b>  return </b></font> 语句来实现这个函数。

> *Note: Remember, <font color="#800080"><b>  uint256 </b></font> is equivalent to <font color="#800080"><b>  uint </b></font>. We've been using <font color="#800080"><b>  uint </b></font> in our code up until now, but we're using <font color="#800080"><b>  uint256 </b></font> here because we copy/pasted from the spec.<br/>注意：要记得， <font color="#800080"><b>  uint256 </b></font> 等同于<font color="#800080"><b>  uint </b></font>。我们从一开始一直在代码中使用 <font color="#800080"><b>  uint </b></font>，但从现在开始我们将在这里用 <font color="#800080"><b>  uint256 </b></font>，因为我们直接从规范中复制粘贴。*

## ERC721: 转移标准 ERC721: Transfer Logic

Now we're going to continue our ERC721 implementation by looking at transfering ownership from one person to another.

现在我们将通过讲解把所有权从一个人转移给另一个人来继续我们的 ERC721 规范的实现。

Note that the ERC721 spec has 2 different ways to transfer tokens:

注意 ERC721 规范有两种不同的方法来转移代币：

```solidity
function transfer(address _to, uint256 _tokenId) public;
```

```solidity
function approve(address _to, uint256 _tokenId) public;
function takeOwnership(uint256 _tokenId) public;
```

  1. The first way is the token's owner calls <font color="#800080"><b>  transferFrom </b></font> with his <font color="#800080"><b>  address </b></font> as the <font color="#800080"><b>  _from </b></font> parameter, the address he wants to transfer to as the<font color="#800080"><b>  _to </b></font>  parameter, and the <font color="#800080"><b>  _tokenId </b></font> of the token he wants to transfer.<br/>第一种方法是代币的拥有者调用<font color="#800080"><b>  transfer </b></font> 方法，传入他想转移到的 font color="#800080"><b>  address </b></font> 和他想转移的代币的 <font color="#800080"><b>  _tokenId </b></font>。

  2. The second way is the token's owner first calls <font color="#800080"><b>  approve </b></font> with the address he wants to transfer to, and the <font color="#800080"><b>  _tokenID </b></font> . The contract then stores who is approved to take a token, usually in a <font color="#800080"><b>  mapping (uint256 => address) </b></font>. Then, when the owner or the approved address calls <font color="#800080"><b>  transferFrom </b></font>, the contract checks if that <font color="#800080"><b>  msg.sender </b></font> is the owner or is approved by the owner to take the token, and if so it transfers the token to him.<br/>第二种方法是代币拥有者首先调用 approve，然后传入与以上相同的参数。接着，该合约会存储谁被允许提取代币，通常存储到一个 <font color="#800080"><b>  mapping (uint256 => address) </b></font> 里。然后，当有人调用 <font color="#800080"><b>  takeOwnership </b></font> 时，合约会检查 <font color="#800080"><b>  msg.sender </b></font> 是否得到拥有者的批准来提取代币，如果是，则将代币转移给他。

Notice that both methods contain the same transfer logic. In one case the sender of the token calls the <font color="#800080"><b>  transferFrom </b></font> function; in the other the owner or the approved receiver of the token calls it.

你注意到了吗，<font color="#800080"><b>  transfer </b></font> 和 <font color="#800080"><b>  takeOwnership </b></font> 都将包含相同的转移逻辑，只是以相反的顺序。 （一种情况是代币的发送者调用函数；另一种情况是代币的接收者调用它）。

So it makes sense for us to abstract this logic into its own private function, <font color="#800080"><b>  _transfer </b></font>, which is then called by <font color="#800080"><b>  transferFrom </b></font>.

所以我们把这个逻辑抽象成它自己的私有函数 <font color="#800080"><b>  _transfer </b></font>，然后由这两个函数来调用它。 这样我们就不用写重复的代码了。

## ERC721: 批准 ERC721: ERC721: Approve

Now, let's implement <font color="#800080"><b>  approve </b></font>.

现在，让我们来实现 <font color="#800080"><b>  approve </b></font>。

Remember, with <font color="#800080"><b>  approve </b></font>the transfer happens in 2 steps:

记住，使用 <font color="#800080"><b>  approve </b></font> 或者 takeOwnership 的时候，转移有2个步骤：

You, the owner, call <font color="#800080"><b>  approve </b></font> and give it the <font color="#800080"><b> _approved </b></font> address of the new owner, and the <font color="#800080"><b> _tokenId </b></font> you want them to take.

你，作为所有者，用新主人的 address 和你希望他获取的 <font color="#800080"><b> _tokenId </b></font> 来调用 <font color="#800080"><b>  approve </b></font>

The new owner calls <font color="#800080"><b> transferFrom </b></font> with the <font color="#800080"><b>  _tokenId </b></font>. Next, the contract checks to make sure the new owner has been already approved, and then transfers them the token.

新主人用 <font color="#800080"><b>  _tokenId </b></font> 来调用 <font color="#800080"><b> transferFrom </b></font>，合约会检查确保他获得了批准，然后把代币转移给他。

Because this happens in 2 function calls, we need to use the  data structure to store who's been approved for what in between function calls.

因为这发生在2个函数的调用中，所以在函数调用之间，我们需要一个数据结构来存储什么人被批准获取什么。

## ERC721: takeOwnership

There is one more thing to do- there's an <font color="#800080"><b>  Approval </b></font> event in the ERC721 spec. So we should fire this event at the end of the <font color="#800080"><b>  approve </b></font> function.

最后一个函数 <font color="#800080"><b>  takeOwnership </b></font>， 应该只是简单地检查以确保 <font color="#800080"><b>  msg.sender </b></font> 已经被批准来提取这个代币。若确认，就调用 <font color="#800080"><b>  _transfer </b></font>；
