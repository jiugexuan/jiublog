---
title: 【ETH】函数修饰符 Function Modifiers
date: 2022-2-22 09:00:00 +/-0800
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

## 函数修饰符 Function Modifiers

A function modifier looks just like a function, but uses the keyword <font color="#800080"><b> modifier </b></font> instead of the keyword <font color="#800080"><b> function </b></font>. And it can't be called directly like a function can — instead we can attach the modifier's name at the end of a function definition to change that function's behavior.

函数修饰符看起来跟函数没什么不同，不过关键字 <font color="#800080"><b> modifier </b></font>告诉编译器，这是个  <font color="#800080"><b> modifier(修饰符) </b></font>，而不是个<font color="#800080"><b> function(函数) </b></font>。它不能像函数那样被直接调用，只能被添加到函数定义的末尾，用以改变函数的行为。

Let's take a closer look by examining <font color="#800080"><b> onlyOwner </b></font>:

咱们仔细读读 <font color="#800080"><b> onlyOwner </b></font>:

```solidity
pragma solidity >=0.5.0 <0.6.0;

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address private _owner;

  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );

  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() internal {
    _owner = msg.sender;
    emit OwnershipTransferred(address(0), _owner);
  }

  /**
   * @return the address of the owner.
   */
  function owner() public view returns(address) {
    return _owner;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(isOwner());
    _;
  }

  /**
   * @return true if `msg.sender` is the owner of the contract.
   */
  function isOwner() public view returns(bool) {
    return msg.sender == _owner;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   * @notice Renouncing to ownership will leave the contract without an owner.
   * It will not be possible to call the functions with the `onlyOwner`
   * modifier anymore.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipTransferred(_owner, address(0));
    _owner = address(0);
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    _transferOwnership(newOwner);
  }

  /**
   * @dev Transfers control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function _transferOwnership(address newOwner) internal {
    require(newOwner != address(0));
    emit OwnershipTransferred(_owner, newOwner);
    _owner = newOwner;
  }
}
```

This is how the onlyOwner function modifier is used:

onlyOwner 函数修饰符是这么用的：

```solidity
contract MyContract is Ownable {
  event LaughManiacally(string laughter);

  // Attention! `onlyOwner` on the field :
  //注意！ `onlyOwner`上场 :
  function likeABoss() external onlyOwner {
    LaughManiacally("Muahahahaha");
  }
}
```

Notice the <font color="#800080"><b> onlyOwner </b></font> modifier on the <font color="#800080"><b> likeABoss </b></font> function. When you call <font color="#800080"><b> likeABoss </b></font>, the code inside <font color="#800080"><b> onlyOwner </b></font> executes first. Then when it hits the <font color="#800080"><b> _ ; </b></font> statement in <font color="#800080"><b> onlyOwner </b></font>, it goes back and executes the code inside <font color="#800080"><b> likeABoss </b></font>.

注意 <font color="#800080"><b> likeABoss </b></font> 函数上的 <font color="#800080"><b> onlyOwner </b></font> 修饰符。 当你调用 <font color="#800080"><b> likeABoss </b></font> 时，首先执行 <font color="#800080"><b> onlyOwner </b></font> 中的代码， 执行到 <font color="#800080"><b> onlyOwner </b></font> 中的 <font color="#800080"><b> _ ; </b></font> 语句时，程序再返回并执行 <font color="#800080"><b> likeABoss </b></font> 中的代码。

So while there are other ways you can use modifiers, one of the most common use-cases is to add a quick <font color="#800080"><b> require </b></font> check before a function executes.

可见，尽管函数修饰符也可以应用到各种场合，但最常见的还是放在函数执行之前添加快速的 <font color="#800080"><b> require </b></font> 检查。

In the case of <font color="#800080"><b> onlyOwner </b></font>, adding this modifier to a function makes it so only the owner of the contract (you, if you deployed it) can call that function.

因为给函数添加了修饰符 <font color="#800080"><b> onlyOwner </b></font>，使得唯有合约的主人（也就是部署者）才能调用它。

> *Note: Giving the owner special powers over the contract like this is often necessary, but it could also be used maliciously. For example, the owner could add a backdoor function that would allow him to transfer anyone's things to himself!<br/>注意：主人对合约享有的特权当然是正当的，不过也可能被恶意使用。比如，万一，主人添加了个后门，允许他偷走别人的东西呢？*

> *So it's important to remember that just because a DApp is on Ethereum does not automatically mean it's decentralized — you have to actually read the full source code to make sure it's free of special controls by the owner that you need to potentially worry about. There's a careful balance as a developer between maintaining control over a DApp such that you can fix potential bugs, and building an owner-less platform that your users can trust to secure their data.<br/>所以非常重要的是，部署在以太坊上的 DApp，并不能保证它真正做到去中心，你需要阅读并理解它的源代码，才能防止其中没有被部署者恶意植入后门；作为开发人员，如何做到既要给自己留下修复 bug 的余地，又要尽量地放权给使用者，以便让他们放心你，从而愿意把数据放在你的 DApp 中，这确实需要个微妙的平衡。*

## 带参数的函数修饰符 Function modifiers with arguments

Previously we looked at the simple example of <font color="#800080"><b> onlyOwner </b></font>. But function modifiers can also take arguments. For example:

之前我们已经读过一个简单的函数修饰符了：<font color="#800080"><b> onlyOwner </b></font>。函数修饰符也可以带参数。例如：

```solidity
// A mapping to store a user's age:
// 存储用户年龄的映射
mapping (uint => uint) public age;

// Modifier that requires this user to be older than a certain age:
// 限定用户年龄的修饰符
modifier olderThan(uint _age, uint _userId) {
  require(age[_userId] >= _age);
  _;
}

// Must be older than 16 to drive a car (in the US, at least).
// We can call the `olderThan` modifier with arguments like so:
// 必须年满16周岁才允许开车 (至少在美国是这样的).
// 我们可以用如下参数调用`olderThan` 修饰符:
function driveCar(uint _userId) public olderThan(16, _userId) {
  // Some function logic
  // 其余的程序逻辑
}
```

You can see here that the <font color="#800080"><b> onlyOwner </b></font> modifier takes arguments just like a function does. And that the <font color="#800080"><b> driveCar </b></font> function passes its arguments to the modifier.

看到了吧， <font color="#800080"><b> onlyOwner </b></font> 修饰符可以像函数一样接收参数，是“宿主”函数 <font color="#800080"><b> driveCar </b></font> 把参数传递给它的修饰符的。
