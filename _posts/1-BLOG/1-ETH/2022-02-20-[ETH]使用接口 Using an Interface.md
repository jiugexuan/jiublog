---
title: 【ETH】使用接口 Using an Interface
date: 2022-2-20 13:00:00 +/-0800
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

## 与其他合约的交互 Interacting with other contracts

For our contract to talk to another contract on the blockchain that we don't own, first we need to define an <b><font color="#0099ff">interface</font></b>.

如果我们的合约需要和区块链上的其他的合约会话，则需先定义一个 <b><font color="#0099ff">interface (接口)</font></b> 。

Let's look at a simple example. Say there was a contract on the blockchain that looked like this:

先举一个简单的栗子。 假设在区块链上有这么一个合约：

```solidity
contract LuckyNumber {
  mapping(address => uint) numbers;

  function setNum(uint _num) public {
    numbers[msg.sender] = _num;
  }

  function getNum(address _myAddress) public view returns (uint) {
    return numbers[_myAddress];
  }
}
```

This would be a simple contract where anyone could store their lucky number, and it will be associated with their Ethereum address. Then anyone else could look up that person's lucky number using their address.

这是个很简单的合约，您可以用它存储自己的幸运号码，并将其与您的以太坊地址关联。 这样其他人就可以通过您的地址查找您的幸运号码了。

Now let's say we had an external contract that wanted to read the data in this contract using the <font color="#800080"><b> getNum </b></font> function.

现在假设我们有一个外部合约，使用 <font color="#800080"><b> getNum </b></font> 函数可读取其中的数据。

First we'd have to define an <b><font color="#0099ff">interface</font></b> of the <font color="#800080"><b> LuckyNumber </b></font> contract:

首先，我们定义 <font color="#800080"><b> LuckyNumber </b></font> 合约的 <b><font color="#0099ff">interface</font></b> ：

```solidity
contract NumberInterface {
  function getNum(address _myAddress) public view returns (uint);
}
```

Notice that this looks like defining a contract, with a few differences. For one, we're only declaring the functions we want to interact with — in this case <font color="#800080"><b> getNum </b></font> — and we don't mention any of the other functions or state variables.

请注意，这个过程虽然看起来像在定义一个合约，但其实内里不同。首先，我们只声明了要与之交互的函数 —— 在本例中为 <font color="#800080"><b> getNum </b></font> —— 在其中我们没有使用到任何其他的函数或状态变量。

Secondly, we're not defining the function bodies. Instead of curly braces (<font color="#800080"><b> { </b></font> and <font color="#800080"><b> } </b></font>), we're simply ending the function declaration with a semi-colon ( </b></font> and <font color="#800080"><b> ; </b></font> ).

其次，我们并没有使用大括号（<font color="#800080"><b> { </b></font> 和 <font color="#800080"><b> } </b></font>）定义函数体，我们单单用分号（</b></font> and <font color="#800080"><b> ; </b></font>）结束了函数声明。这使它看起来像一个合约框架。

So it kind of looks like a contract skeleton. This is how the compiler knows it's an interface.

编译器就是靠这些特征认出它是一个接口的。

By including this interface in our dapp's code our contract knows what the other contract's functions look like, how to call them, and what sort of response to expect.

在我们的 app 代码中使用这个接口，合约就知道其他合约的函数是怎样的，应该如何调用，以及可期待什么类型的返回值

## 使用接口 Using an Interface

Continuing our previous example with <font color="#800080"><b>NumberInterface </b></font>, once we've defined the interface as:

继续前面 <font color="#800080"><b>NumberInterface </b></font> 的例子，我们既然将接口定义为：

```solidity
contract NumberInterface {
  function getNum(address _myAddress) public view returns (uint);
}
```


We can use it in a contract as follows:

我们可以在合约中这样使用：

```solidity
contract MyContract {
  address NumberInterfaceAddress = 0xab38...;
  // ^ The address of the FavoriteNumber contract on Ethereum
  // ^ 这是FavoriteNumber合约在以太坊上的地址
  NumberInterface numberContract = NumberInterface(NumberInterfaceAddress);
  // Now `numberContract` is pointing to the other contract
  // 现在变量 `numberContract` 指向另一个合约对象

  function someFunction() public {
    // Now we can call `getNum` from that contract:
    // 现在我们可以调用在那个合约中声明的 `getNum`函数:
    uint num = numberContract.getNum(msg.sender);
    // ...and do something with `num` here
    // ...在这儿使用 `num`变量做些什么
  }
}
```


In this way, your contract can interact with any other contract on the Ethereum blockchain, as long they expose those functions as <font color="#800080"><b>public </b></font> or <font color="#800080"><b>external </b></font>external.

通过这种方式，只要将您合约的可见性设置为 <font color="#800080"><b>public(公共) </b></font>或<font color="#800080"><b>external(外部) </b></font>，它们就可以与以太坊区块链上的任何其他合约进行交互。
