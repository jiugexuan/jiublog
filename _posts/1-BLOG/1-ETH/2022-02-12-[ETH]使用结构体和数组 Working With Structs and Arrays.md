---
title: 【ETH】使用结构体和数组 Working With Structs and Arrays
date: 2022-2-12 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---


Remember our <font color=purple><b> Person </b></font> struct in the previous example?
还记得 <font color=purple><b> Person </b></font> 结构吗？


```solidity
struct Person {
  uint age;
  string name;
}

Person[] public people;
```
Now we're going to learn how to create new <font color=purple><b> Person </b></font> and add them to our <font color=purple><b> people </b></font> array.

现在我们创建新的 <font color=purple><b> Person </b></font> 结构，然后把它加入到名为 <font color=purple><b> people </b></font> 的数组中.


```solidity
// create a New Person:
// 创建一个新的Person:
Person satoshi = Person(172, "Satoshi");

// Add that person to the Array:
// 将新创建的satoshi添加进people数组:
people.push(satoshi);

```

We can also combine these together and do them in one line of code to keep things clean:

你也可以两步并一步，用一行代码更简洁:

```solidty
people.push(Person(16, "Jack"));
```
<br/>

> *Note that <font color=purple> array.push() </font> adds something to the end of the array, so the elements are in the order we added them. See the following example:<br/>注：<font color=purple> array.push() </font>在数组的 尾部 加入新元素 ，所以元素在数组中的顺序就是我们添加的顺序， 如:*

```solidty
uint[] numbers;
numbers.push(5);
numbers.push(10);
numbers.push(15);
// numbers is now equal to [5, 10, 15]
}
```
