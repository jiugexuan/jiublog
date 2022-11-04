---
title: 【ETH】私有/公共函数 Private / Public Functions
date: 2022-2-13 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

In Solidity, functions are <font color=purple><b> public </b></font> by default. This means anyone (or any other contract) can call your contract's function and execute its code.

Solidity 定义的函数的属性默认为<font color=purple><b> 公共</b></font>。这就意味着任何一方 (或其它合约) 都可以调用你合约里的函数。

Obviously this isn't always desirable, and can make your contract vulnerable to attacks. Thus it's good practice to mark your functions as <font color=purple><b> private </b></font> by default, and then only make <font color=purple><b> public </b></font> the functions you want to expose to the world.

显然，不是什么时候都需要这样，而且这样的合约易于受到攻击。 所以将自己的函数定义为<font color=purple><b> 私有 </b></font>私有是一个好的编程习惯，只有当你需要外部世界调用它时才将它设置为<font color=purple><b> 公共 </b></font>公共。

Let's look at how to declare a private function:
如何定义一个私有的函数呢？


```solidity
uint[] numbers;

function _addToArray(uint _number) private {
  numbers.push(_number);
}
```
This means only other functions within our contract will be able to call this function and add to the <font color=purple><b> numbers </b></font> array.

这意味着只有我们合约中的其它函数才能够调用这个函数，给 <font color=purple><b> numbers </b></font> 数组添加新成员。

As you can see, we use the keyword <font color=purple><b> private </b></font> after the function name. And as with function parameters, it's convention to start private function names with an underscore ( <font color=purple><b> _ </b></font> ).

可以看到，在函数名字后面使用关键字 <font color=purple><b> private </b></font> 即可。和函数的参数类似，私有函数的名字用( <font color=purple><b> _ </b></font> )起始。
