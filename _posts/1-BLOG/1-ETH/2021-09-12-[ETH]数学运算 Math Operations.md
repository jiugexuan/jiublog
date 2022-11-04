---
title: 【ETH】数学运算 Math Operations
date: 2021-09-12 20:00:00 +/-0800
categories: [BLOG,ETH]
tags: [ETH]     # TAG names should always be lowercase 标记名称应始终为小写
---

Math in Solidity is pretty straightforward. The following operations are the same as in most programming languages:

在 Solidity 中，数学运算很直观明了，与其它程序设计语言相同:

Addition: <font color=purple><b> x + y </b></font><br/>
Subtraction: <font color=purple><b> x - y </b></font><br/>
Multiplication: <font color=purple><b> x * y </b></font><br/>
Division: <font color=purple><b> x / y </b></font><br/>
Modulus / remainder: <font color=purple><b> x % y </b></font> (for example, <font color=purple><b> 13 % 5 </b></font> is <font color=purple><b> 3 </b></font>, because if you divide 5 into 13, 3 is the remainder)<br/>

加法: <font color=purple><b> x + y </b></font><br/>
减法: <font color=purple><b> x - y </b></font><br/>
乘法: <font color=purple><b> x * y </b></font><br/>
除法: <font color=purple><b> x / y </b></font><br/>
取模 / 求余: <font color=purple><b> x % y </b></font> (例如, <font color=purple><b> 13 % 5 </b></font><b> 余 </b><font color=purple><b> 3 </b></font>, 因为13除以5，余3)<br/>

Solidity also supports <font color="#0099ff"><b>exponential operator</b></font> an  (i.e. "x to the power of y", x^y):

Solidity 还支持 <font color="#0099ff"><b>乘方操作</b></font> (如：x 的 y次方） // 例如： 5 ** 2 = 25

```solidity
uint x = 5 ** 2; // equal to 5^2 = 25
```
