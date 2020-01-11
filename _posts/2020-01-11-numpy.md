---
titile: "numpy笔记"
categories:
  - Numpy
tags:
  - Python
  - 数据分析
  - 数据结构
---

#### import numpy as np
# 数组的创建
    np.array()
## 通过重复数组来构建新数组
    np.tile(arr, reps)
        :arr: 需要重复的数组
        :reps: 重复的次数
## 通过可迭代对象构建数组
    np.fromiter(iterable, dtype, count)
        :count: 读取的可迭代元素数量，默认-1，读取全部

# 高级索引
    x = [[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8],
         [ 9, 10, 11]]
## 整数数组索引
    # 索引数组的对角元素
    rows = np.array([[0,0], [3,3]])
    cols = np.array([[0,2], [0,2]])
    y = x[rows, cols] 
## 布尔索引
    x = np.array([np.nan, 2,3,4, np.nan])
    x[~np.isnan(x)]     # ~表示取反运算符
## 花式索引
    h = x[[3,1,2]]
    [[ 9, 10, 11],
     [ 3,  4,  5],
     [ 6,  7,  8]]
    以索引数组的值为目标数组的行下标，将整行复制到新的数组
## 元素在数组中的原始索引
    np.unravel_index(indices, shape)
        :indices: 整型数值或整型数组
        :shape: 原始数组的形状

# broadcast(广播)
对不同形状的数组进行数值计算的方式

# 迭代数组
迭代数组的元素访问顺序是元素在内存中的存储顺序，默认为行序优先
## 遍历数组
    for i in np.nditer(a, order="C", op_flags=['readwrite'], flags=['external_loop']):
        print(i)

    # op_flags: 指定元素的读写权限
    # order: 指定遍历的顺序，默认行序（C）
    # flags: 控制迭代器行为
        c_index: 跟踪C顺序的索引
        f_index: 跟踪F顺序的索引
        multi_index: 每次迭代可以跟踪一种索引类型
        external_loop: 结果是一个一维数组

## 广播迭代
如果两个数组是可迭代的，nditer可以组合对象同时迭代他们

# 操作数组
## 移动数组的轴
    # 移动axis轴到start的位置，可以通过shape来查看移动情况
    np.moveaxis(ndarray, axis, start)

## 两轴互换位置
    np.swapaxes(ndarray, axis1, axis2)

## 产生模仿广播的对象，返回值为迭代器
    np.broadcast(x,y)

## 将数组重新广播为一个新的形状
    np.broadcast_to(ndarray, shape, subok)
        :subok: 如果为True，子类将被传递，否则返回的数组将被强制为基类数组

## 指定位置插入新的轴来扩展数组形状
    np.expand_dims(ndarray, axis)

## 连接两个数组
    np.concatenate((a1,a2,...), axis=0)
        :axis: 表示连接方向

## 堆叠数组
### 沿轴堆叠两个数组,形成三维数组
    np.stack((a1,a2), axis=0)

### 水平堆叠数组
    np.hstack((a1,b1))

### 垂直堆叠
    np.vstack((a1,b1))

## 分割数组
    np.split(arr, indices_or_sections, axis)
        :indices_or_sections: 如果是整数，表示将数组平均切分，如果是数组，表示切分的位置
        :axis: 切割维度
### 水平切割数组(列切割)

    # hsplit相当于split的indices_or_sections=1的情况
    np.hsplit(arr,indices_or_sections)
        :indices_or_sections: 切割后的数组数量。必须能被原数组的列数整除
### 垂直切割数组(行切割)
    np.vsplit(arr, indices_or_sections)

## 重塑数组
    arr.reshape(shape)
        :shape: 重塑后的形状,shape的各轴乘积必须等于数组元素之和
    
    np.resize(arr, shape)
    # 允许重塑后的大小大于原始大小，多出来的元素由原数组的元素依次填充

# 操作数组的元素
## 追加元素
    # 会分配新的数组 空间复杂度O(n)
    np.append(arr, values, axis)
        :axis: 为None时表示横向追加，返回始终为一维数组；为1时表示追加列，行数必须相等；为0时表示追加行，列数必须相等。
## 插入元素
    np.insert(arr, obj, values, axis)
        :obj: 插入位置的索引
        :values: 插入的值，可以是单值或数组
        :axis: 以轴为索引插入值，如果为None，返回值为一维数组

## 删除元素
    np.delete(arr, obj, axis)
        :obj: 删除的元素，单值或数组
        :axis: 为None时返回一维数组

# 数组的计算
## 数学函数
### 正弦
    np.sin(30*np.pi/180)    # 0.5
### 余弦
    np.cos(60*np.pi/180)    # 0.5
### 正切
    np.tan(45*np.pi/180)    # 1
## 取整
    np.around(a, decimal)
        :decimal: 舍入的小数位数
    
    # 向下取整
    np.floor()

    # 向上取整
    np.ceil()

    # 取整到最接近的整数
    np.rint()
## 算术
    np.add(a,b)             # 加
    np.subtract(a,b)        # 减
    np.multiply(a,b)        # 乘
    np.divide(a,b)          # 除
    np.reciprocal(a)        # 倒数
    np.power(a,b)           # a的b次幂
    np.mod(a,b)             # 模(取余)
    np.remainder(a,b)       # 取余
## 统计
    np.amax(arr)                        # 最大值
    np.amin(arr)                        # 最小值
    np.ptp(arr, axis)                   # 最大最小差值
        :axis: None表示全部元素；0表示返回每列元素的差值；1表示返回每行元素的差值
    np.percentile(arr, q, axis)         # 百分位数
        :q: 要计算的百分位数
        :axis: None表示全部元素求百分位数；0表示每列；1表示每行
    np.median(arr, axis)                # 中位数
        :axis: 同上
    np.mean(arr, axis)                  # 算数平均
    np.average(arr, weights, axis)      # 加权平均
        :weights: 权重
## 数组间的计算
### 点积
    # 如果两个都是二维数组，建议使用np.matmul或者a@b
    np.dot(a, b, out=None)
### 数组的交集
    np.intersect1d(arr1, arr2)

# 副本和视图
## 赋值
简单的赋值，使用的是相同的地址，修改或改变形状会导致指向地址的所有变量都改变
## 视图(浅拷贝)
视图与原数组指向的地址不同，修改视图形状不会改变原数组，但修改数据会
## 副本(深拷贝)
指向的地址不同，不会影响原数组

# 关闭警告提示
    with np.errstate(all="ignore"):
        np.ones(1) / 0
# 日期
    np.datetime64(time, units)
        :time: 可以是字符串形式的'today'等，或者是timedeltas
        :units: 日期单位
            D: 天
            Y: 年
            M: 月
            W: 周

    np.timedelta64()

## 工作日相关函数
### 获取工作日
    np.busday_offset(date, offset, roll)
        :offset: 偏移量，指定输入date的偏移量，以天为单位
        :roll: 偏移的动作，向前或向后偏移，只有当输入date处于非工作日时有效
### 是否工作日
    np.is_busday(date)
### 工作日统计
    np.busday_count(date1, date2)