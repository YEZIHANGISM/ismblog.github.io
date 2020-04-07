---
titile: "数据处理"
categories:
  - Numpy
  - Pandas
tags:
  - numpy
  - pandas
---

# Numpy
## 数组
```np.array()```

### 通过重复数组来构建新数组
```python
np.tile(arr, reps)
    # arr: 需要重复的数组
    # reps: 重复的次数
```
        
### 通过可迭代对象构建数组
```python
np.fromiter(iterable, dtype, count)
    # count: 读取的可迭代元素数量，默认-1，读取全部
```
## 数组索引
### 高级索引
    x = [[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8],
         [ 9, 10, 11]]

### 整数数组索引
    # 索引数组的对角元素
    rows = np.array([[0,0], [3,3]])
    cols = np.array([[0,2], [0,2]])
    y = x[rows, cols] 

### 布尔索引
    x = np.array([np.nan, 2,3,4, np.nan])
    x[~np.isnan(x)]     # ~表示取反运算符

### 花式索引
    h = x[[3,1,2]]
    [[ 9, 10, 11],
     [ 3,  4,  5],
     [ 6,  7,  8]]
    以索引数组的值为目标数组的行下标，将整行复制到新的数组

### 元素在数组中的原始索引
    np.unravel_index(indices, shape)
        :indices: 整型数值或整型数组
        :shape: 原始数组的形状

## broadcast(广播)
对不同形状的数组进行数值计算的方式

## 数组迭代
迭代数组的元素访问顺序是元素在内存中的存储顺序，默认为行序优先

### 遍历数组
```python
for i in np.nditer(a, order="C", op_flags=['readwrite'], flags=['external_loop']):
    print(i)
```

参数详解:
- op_flags: 指定元素的读写权限
- order: 指定遍历的顺序，默认行序（C）
- flags: 控制迭代器行为
    - c_index: 跟踪C顺序的索引
    - f_index: 跟踪F顺序的索引
    - multi_index: 每次迭代可以跟踪一种索引类型
    - external_loop: 结果是一个一维数组

### 广播迭代
如果两个数组是可迭代的，nditer可以组合对象同时迭代他们

## 操作数组
### 移动数组的轴
```python
# 移动axis轴到start的位置，可以通过shape来查看移动情况
np.moveaxis(ndarray, axis, start)
```

### 两轴互换位置
```python
np.swapaxes(ndarray, axis1, axis2)
```

### 产生模仿广播的对象，返回值为迭代器
```python
np.broadcast(x,y)
```

### 将数组重新广播为一个新的形状
```python
np.broadcast_to(ndarray, shape, subok)
    # subok: 如果为True，子类将被传递，否则返回的数组将被强制为基类数组
```

### 指定位置插入新的轴来扩展数组形状
```python
np.expand_dims(ndarray, axis)
```

### 连接两个数组
```python
np.concatenate((a1,a2,...), axis=0)
    # axis: 表示连接方向
```

### 堆叠数组
沿轴堆叠两个数组,形成三维数组
```python
np.stack((a1,a2), axis=0)
```

水平堆叠数组
```python
np.hstack((a1,b1))
```

垂直堆叠
```python
np.vstack((a1,b1))
```

### 分割数组
```python
np.split(arr, indices_or_sections, axis)
    # indices_or_sections: 如果是整数，表示将数组平均切分，如果是数组，表示切分的位置
    # axis: 切割维度
```

水平切割数组(列切割)
```python
# hsplit相当于split的indices_or_sections=1的情况
np.hsplit(arr,indices_or_sections)
    # indices_or_sections: 切割后的数组数量。必须能被原数组的列数整除
```

垂直切割数组(行切割)
```python
np.vsplit(arr, indices_or_sections)
```

### 重塑数组
```python
arr.reshape(shape)
    # shape: 重塑后的形状,shape的各轴乘积必须等于数组元素之和

np.resize(arr, shape)
# 允许重塑后的大小大于原始大小，多出来的元素由原数组的元素依次填充
```

## 操作数组的元素
### 追加元素
```python
# 会分配新的数组 空间复杂度O(n)
np.append(arr, values, axis)
```

- axis: 为None时表示横向追加，返回始终为一维数组；为1时表示追加列，行数必须相等；为0时表示追加行，列数必须相等。

### 插入元素
```python
np.insert(arr, obj, values, axis)
```

- obj: 插入位置的索引
- values: 插入的值，可以是单值或数组
- axis: 以轴为索引插入值，如果为None，返回值为一维数组

### 删除元素
```python
np.delete(arr, obj, axis)
```

- obj: 删除的元素，单值或数组
- axis: 为None时返回一维数组

## 数组的计算
### 数学函数
- 正弦 - `np.sin(30*np.pi/180)    # 0.5`
- 余弦 - `np.cos(60*np.pi/180)    # 0.5`
- 正切 - `np.tan(45*np.pi/180)    # 1`

### 取整
```python
np.around(a, decimal)
    # decimal: 舍入的小数位数

# 向下取整
np.floor()

# 向上取整
np.ceil()

# 取整到最接近的整数
np.rint()
```

### 算术
```python
np.add(a,b)             # 加
np.subtract(a,b)        # 减
np.multiply(a,b)        # 乘
np.divide(a,b)          # 除
np.reciprocal(a)        # 倒数
np.power(a,b)           # a的b次幂
np.mod(a,b)             # 模(取余)
np.remainder(a,b)       # 取余
```

### 统计
```python
np.amax(arr)                        # 最大值
np.amin(arr)                        # 最小值
np.ptp(arr, axis)                   # 最大最小差值
    # axis: None表示全部元素；0表示返回每列元素的差值；1表示返回每行元素的差值

np.percentile(arr, q, axis)         # 百分位数
    # q: 要计算的百分位数
    # axis: None表示全部元素求百分位数；0表示每列；1表示每行

np.median(arr, axis)                # 中位数
    # axis: 同上

np.mean(arr, axis)                  # 算数平均
np.average(arr, weights, axis)      # 加权平均
    # weights: 权重
```

### 数组间的计算
点积

> 如果两个都是二维数组，建议使用np.matmul或者a@b
```python
np.dot(a, b, out=None)
```

交集
```python
np.intersect1d(arr1, arr2)
```

## 矩阵
`import numpy.matlib`
- 初始化新的矩阵 - `np.matlib.empty(shape, dtype, order)`
- 初始化以0填充的矩阵 - `np.matlib.zeros(shape)`
- 初始化以1填充的矩阵 - `np.matlib.ones(shape)`
- 初始化对角元素为1的矩阵 - `np.matlib.eye(n, m, k, dtype)`
    - n - 矩阵行数
    - m - 矩阵列数
    - k - 对角线索引
- 初始化单位矩阵 - `np.matlib.identity(n, dtype)`
    - n - 矩阵行列数
- 初始化随机数值的矩阵 - `np.matlib.rand(n, m)`
    - n - 行数
    - m - 列数

### 矩阵运算
- 迹 - `np.trace(matrix)`
- 矩阵乘积 - `np.matmul(x1, x2)` 或者 `x1@x2`

## 副本和视图
### 赋值
简单的赋值，使用的是相同的地址，修改或改变形状会导致指向地址的所有变量都改变

### 视图(浅拷贝)
视图与原数组指向的地址不同，修改视图形状不会改变原数组，但修改数据会

### 副本(深拷贝)
指向的地址不同，不会影响原数组

## 关闭警告提示
```python
with np.errstate(all="ignore"):
    np.ones(1) / 0
```

# 日期
```python
np.datetime64(time, units)
    :time: 可以是字符串形式的'today'等，或者是timedeltas
    :units: 日期单位
        D: 天
        Y: 年
        M: 月
        W: 周

np.timedelta64()
```

## 工作日相关函数
### 获取工作日
```python
np.busday_offset(date, offset, roll)
    offset: 偏移量，指定输入date的偏移量，以天为单位
    roll: 偏移的动作，向前或向后偏移，只有当输入date处于非工作日时有效
```

### 是否工作日
```python
np.is_busday(date)
```

### 工作日统计
```python
np.busday_count(date1, date2)
```

# Pandas
## DataFrame
```python
pandas.DataFrame(data, index, columns, dtype, copy)
```

参数详解：
- data - 数据来源。可以是*ndarray*、*series*、*map*、*dict*、*lists*、*constant*或者另一个*DataFrame*
- index - 行标签。如果不指定，默认为`np.arrange(n)`
- columns - 列标签。如果不指定，默认为`np.arrange(n)`
- dtype - 每一列的数据类型
- copy - 复制数据

### 创建空的DataFrame
```python
import pandas as pd

df = pd.DataFrame()
print(df)
```

### 从列表创建DataFrame
```python
data = [i for i in range(5)]
df = pd.DataFrame(data)
print(df)
```

列表的形式可以是多样的。可以是单列表，也可以是嵌套列表。或者字典列表等
```python
data = [["a",10], ["b",12], ["v",14]]
# 将得到一个3*2矩阵

data = [{"a":1, "b":2},{"a":3, "b":4, "c":12}]
# 将得到一个2*3矩阵，字典的key默认为列名，缺少的值补NaN
# 如果指定的列标签在字典中不存在，则数值全部为NaN
```

### 从字典创建DataFrame
```python
data = {"name":['a','b','c'], "age":[12,32,11]}
# 将生成一个3*2矩阵，列名为字典的key
```

### 添加列
```python
data = {
    "a":[i for i in range(4)],
    "b":[i for i in range(3,7)]
    }
df = pd.DataFrame(data)
df["c"] = [i for i in range(2,6)]
df["d"] = df["a"]-df["b"]
```

### 删除列
```python
def df["a"]

df.pop["b"]
```

### 选择行
```python
df.loc["b"]
# 按默认的行索引选择

df.iloc[2]
# 按整数位置选择
```

### 添加行
```python
df = pd.DataFrame([[1,2], [3,4]], columns=list("ab"))

df.loc[4] = [i for i in range(5,9)]
# 单行添加

df2 = pd.DataFrame([[5,6], [7,8]], columns=list("ab"))
df = df.append(df2, ignore_index=True)
# 合并两个DataFrame
# ignore_index避免合并后的行索引重复。
```

### 删除行
```python
df.drop(0)
```

### 选择特定的值
```python
df.at[column, index]
```

### 常用方法
- df.T - 转置
- df.axes - 返回行列标签
- df.empty - 判断是否为控
- df.ndmin - 返回维度
- df.dtypes - 返回每一列的数据类型
- df.shape - 返回维度元组
- df.size - 返回元素数
- df.values - 返回所有数据，类型为*ndarray*
- df.head(n) - 返回前N行，默认为5
- df.tail(n) - 返回后N行，默认为5
- df.sum(axis) - 对数据进行求和，axis默认为0，表示对列求和，返回一个序列；axis为1表示对行进行求和，自动忽略不能相加的数据。
- df.mean(axis) - 求平均值。axis同上。
- df.describe() - 计算数据的统计信息摘要
- **df.pipe(func)** - 将func函数作用在整个DataFrame
- **df.apply(func, axis)** - 将func函数作用在单独的行列上，axis同上。
- **df.applymap(func)** - 将func函数作用在每一个元素上。
- df.reindex(index, columns) - 重建索引，不存在的列将填充NaN
- df.reindex_like(dataframe, method, limit) - 与其他数据帧对齐索引
    - method - 填充方法，
        - pad/ffill - 向前填充
        - bfill/backfill - 向后填充
        - nearest - 从最近的索引值填充
    - limit - 限制填充的最大计数
- df.rename(columns, index) - 重命名轴，参数为字典形式

### 迭代
根据迭代的数据类型不同，产生不同的迭代值：
- series - 值
- DataFrame - 列标签
- Panel - 项目标签

遍历列
```python
df = pd.DataFrame(np.random.randn(3,4))
for col in df:
    print(col)
    # 0
    # 1
    # ...
```

遍历行
```python
for k,v in df.iteritems():
    # k为列索引，v为值

for row_idx,row in df.iterrows():
    # row_idx为行索引，row为每一列与其对应的值

for row in df.itertuples():
    # row表示每一行的对象
    # 返回的是原始对象的副本
```

### 排序
按标签排序
```python
df.sort_index(axis, ascending)
# axis默认为0，按照行排序
```

按值排序
```python
df = df.sort_values(by="col1", kind='mergesort')
# by指定列
# kind指定排序算法
```