---
title: 【ETH】定义函数 Function Declarations
date: 2022-2-11 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

A function declaration in solidity looks like the following

在 Solidity 中函数定义的句法如下:

```solidity
function helloWorld(string _name, uint _amount) {

}
```

This is a function named <font color=purple><b> helloWorld </b></font> that takes 2 parameters: a <font color=purple><b> string </b></font> and a <font color=purple><b>unit </b></font>. For now the body of the function is empty. Note that we're specifying the function visibility as public. We're also providing instructions about where the _name variable should be stored- in memory. This is required for all reference types such as arrays, structs, mappings, and strings.

这是一个名为 <font color=purple><b> helloWorld </b></font> 的函数，它接受两个参数：一个 <font color=purple><b> string </b></font> 类型的 和 一个 <font color=purple><b> unit </b></font> 类型的。现在函数内部还是空的。我们还提供了关于 _name 变量应该存储在内存中的位置的说明。 这对于所有引用类型（例如数组、结构、映射和字符串）都是必需的。

Well, there are two ways in which you can pass an argument to a Solidity function:
有两种方法可以将参数传递给 Solidity 函数：

1. By value, which means that the Solidity compiler creates a new copy of the parameter's value and passes it to your function. This allows your function to modify the value without worrying that the value of the initial parameter gets changed.<br/>按值，这意味着 Solidity 编译器创建参数值的新副本并将其传递给您的函数。 这允许您的函数修改该值，而不必担心初始参数的值会被更改。

2. By reference, which means that your function is called with a... reference to the original variable. Thus, if your function changes the value of the variable it receives, the value of the original variable gets changed.<br/>通过引用，这意味着您的函数是使用...引用原始变量来调用的。 因此，如果你的函数改变了它接收到的变量的值，那么原始变量的值就会改变。



> *Note: It's convention (but not required) to start function parameter variable names with an underscore (<font color=purple> <b> _ </b> </font> ) in order to differentiate them from global variables. We'll use that convention throughout our tutorial.
注：习惯上函数里的变量都是以( <font color=purple><b> _ </b></font> )开头 (但不是硬性规定) 以区别全局变量。*
