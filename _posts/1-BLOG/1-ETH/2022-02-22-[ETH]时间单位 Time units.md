---
title: 【ETH】时间单位 Time units
date: 2022-2-22 11:00:00 +/-0800
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

Solidity provides some native units for dealing with time.

Solidity 使用自己的本地时间单位。

The variable <font color="#800080"><b> now </b></font> will return the current unix timestamp of the latest block (the number of seconds that have passed since January 1st 1970).

变量 <font color="#800080"><b> now </b></font> 将返回当前的unix时间戳（自1970年1月1日以来经过的秒数）。

> *Note: Unix time is traditionally stored in a 32-bit number. This will lead to the "Year 2038" problem, when 32-bit unix timestamps will overflow and break a lot of legacy systems. So if we wanted our DApp to keep running 20 years from now, we could use a 64-bit number instead — but our users would have to spend more gas to use our DApp in the meantime. Design decisions!<br/>注意：Unix时间传统用一个32位的整数进行存储。这会导致“2038年”问题，当这个32位的unix时间戳不够用，产生溢出，使用这个时间的遗留系统就麻烦了。所以，如果我们想让我们的 DApp 跑够20年，我们可以使用64位整数表示时间，但为此我们的用户又得支付更多的 gas。真是个两难的设计啊！*

Solidity also contains the time units <font color="#800080"><b> seconds </b></font>, <font color="#800080"><b> minutes </b></font>, <font color="#800080"><b> hours </b></font>, <font color="#800080"><b> days </b></font>, <font color="#800080"><b> weeks </b></font> and <font color="#800080"><b> years </b></font>. These will convert to a <font color="#800080"><b> uint </b></font> of the number of seconds in that length of time. So <font color="#800080"><b> 1 </b></font> <font color="#800080"><b> minutes </b></font> is <font color="#800080"><b> 60 </b></font>, <font color="#800080"><b> 1 </b></font> <font color="#800080"><b> hours </b></font> is <font color="#800080"><b> 3600  </b></font>(60 seconds x 60 minutes), <font color="#800080"><b> 1 </b></font> <font color="#800080"><b> days </b></font> is <font color="#800080"><b> 86400 </b></font> (24 hours x 60 minutes x 60 seconds), etc.

Solidity 还包含<font color="#800080"><b> 秒(seconds) </b></font>，<font color="#800080"><b> 分钟(minutes) </b></font>，<font color="#800080"><b> 小时(hours) </b></font>，<font color="#800080"><b> 天(days) </b></font>，<font color="#800080"><b> 周(weeks)  </b></font>和 <font color="#800080"><b> 年(years)  </b></font> 等时间单位。它们都会转换成对应的秒数放入 uint 中。所以 <font color="#800080"><b> 1分钟 </b></font> 就是 <font color="#800080"><b> 60 </b></font>，<font color="#800080"><b> 1小时 </b></font>是 <font color="#800080"><b> 3600 </b></font>（60秒×60分钟），<font color="#800080"><b> 1天 </b></font>是<font color="#800080"><b> 86400 </b></font>（24小时×60分钟×60秒），以此类推。

Here's an example of how these time units can be useful:

下面是一些使用时间单位的实用案例：

```solidity

uint lastUpdated;

// Set `lastUpdated` to `now`
// 将‘上次更新时间’ 设置为 ‘现在’
function updateTimestamp() public {
  lastUpdated = now;
}

// Will return `true` if 5 minutes have passed since `updateTimestamp` was
// called, `false` if 5 minutes have not passed
// 如果到上次`updateTimestamp` 超过5分钟，返回 'true'
// 不到5分钟返回 'false'
function fiveMinutesHavePassed() public view returns (bool) {
  return (now >= (lastUpdated + 5 minutes));
}

```
