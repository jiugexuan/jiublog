---
title: 【ETH】OpenZeppelin
date: 2022-2-21 13:00:00 +/-0800
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

OpenZeppelin’s smart contract repository is an invaluable resource for Ethereum developers. It has community-vetted implementations of ERC token standards, security protocols, and other utilities that enable developers to focus on functionality, reducing the need to reinvent the wheel.

OpenZeppelin的智能合约代码库是以太坊开发者的宝库，OpenZeppelin代码库包含了经过社区审查的ERC代币标准、安全协议以及很多的辅助工具库，这些代码可以帮助开发者专注业务逻辑的，而无需重新发明轮子。


## 访问控制合约 Controlling Access

### OpenZeppelin's Ownable contract

Below is the <font color="#800080"><b> Ownable </b></font> contract taken from the <b><font color="#0099ff">OpenZeppelin</font></b> Solidity library. OpenZeppelin is a library of secure and community-vetted smart contracts that you can use in your own DApps.

下面是一个 <font color="#800080"><b> Ownable </b></font> 合约的例子： 来自 <b><font color="#0099ff">OpenZeppelin</font></b> Solidity 库的 Ownable 合约。 OpenZeppelin 是主打安保和社区审查的智能合约库，您可以在自己的 DApps中引用。

```solidity
/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  function Ownable() public {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
}
```

 - Constructors: <font color="#800080"><b> constructor() </b></font> is a <b><font color="#0099ff">constructor</font></b>, which is an optional special function that has the same name as the contract. It will get executed only one time, when the contract is first created.<br/>构造函数：<font color="#800080"><b> function Ownable() </b></font>是一个 <b><font color="#0099ff"> constructor(构造函数)</font></b>，构造函数不是必须的，它与合约同名，构造函数一生中唯一的一次执行，就是在合约最初被创建的时候。

 - Function Modifiers: <font color="#800080"><b> modifier onlyOwner() </b></font>. Modifiers are kind of half-functions that are used to modify other functions, usually to check some requirements prior to execution. In this case, <font color="#800080"><b> onlyOwner </b></font> can be used to limit access so only the owner of the contract can run this function. We'll talk more about function modifiers in the next chapter, and what that weird <font color="#800080"><b> _ ; </b></font> does.<br/>函数修饰符：<font color="#800080"><b> modifier onlyOwner() </b></font>。 修饰符跟函数很类似，不过是用来修饰其他已有函数用的， 在其他语句执行前，为它检查下先验条件。 在这个例子中，我们就可以写个修饰符 <font color="#800080"><b> onlyOwner </b></font> 检查下调用者，确保只有合约的主人才能运行本函数。我们下一章中会详细讲述修饰符，以及那个奇怪的<font color="#800080"><b> _ ; </b></font>。

 - <font color="#800080"><b>indexed </b></font> keyword: don't worry about this one, we don't need it yet.<br/><font color="#800080"><b>indexed </b></font> 关键字：别担心，我们还用不到它。

So the <font color="#800080"><b>Ownable </b></font> contract basically does the following:

所以 <font color="#800080"><b>Ownable </b></font> 合约基本都会这么干：

  1. When a contract is created, its constructor sets the <font color="#800080"><b> owner </b></font>  to <font color="#800080"><b> msg.sender </b></font> (the person who deployed it).<br/>合约创建，构造函数先行，将其 <font color="#800080"><b> owner </b></font> 设置为<font color="#800080"><b> msg.sender </b></font>（其部署者）

  2. It adds an <font color="#800080"><b> onlyOwner </b></font> modifier, which can restrict access to certain functions to only the <font color="#800080"><b> owner </b></font><br/>为它加上一个修饰符 <font color="#800080"><b> onlyOwner </b></font>，它会限制陌生人的访问，将访问某些函数的权限锁定在 <font color="#800080"><b> owner </b></font> 上。

  3. It allows you to transfer the contract to a new <font color="#800080"><b> owner </b></font>.<br/>允许将合约<font color="#800080"><b> 所有权 </b></font>转让给他人。

<font color="#800080"><b>onlyOwner </b></font>  is such a common requirement for contracts that most Solidity DApps start with a copy/paste of this <font color="#800080"><b>Ownable </b></font> contract, and then their first contract inherits from it.

<font color="#800080"><b>onlyOwner </b></font> 简直人见人爱，大多数人开发自己的 Solidity DApps，都是从复制/粘贴 <font color="#800080"><b>Ownable </b></font> 开始的，从它再继承出的子类，并在之上进行功能开发。
