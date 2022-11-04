---
title: 【ETH】结构体 Structs
date: 2021-09-13 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

Sometimes you need a more complex data type. For this, Solidity provides <b><font color="#0099ff">structs</font></b>:

有时你需要更复杂的数据类型，Solidity 提供了 <b><font color="#0099ff">结构体</font></b>
```solidity
struct Person {
  uint age;
  string name;
}
```

Structs allow you to create more complicated data types that have multiple properties.

> *Note that we just introduced a new type, string. Strings are used for arbitrary-length UTF-8 data. Ex. <font color=purple> string greeting = "Hello world!" </font>*

结构体允许你生成一个更复杂的数据类型，它有多个属性。

> *注：我们刚刚引进了一个新类型, string。 字符串用于保存任意长度的 UTF-8 编码数据。 如： <font color=purple> string greeting = "Hello world!" </font>。*
