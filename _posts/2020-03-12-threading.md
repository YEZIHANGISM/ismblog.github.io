---
title: "多线程"
categoies:
  - Python
tags:
  - threading
---

# `threading.Thread`
线程对象，通过该对象创建线程。用户可以继承该对象以实现自定义的线程类。也可以传递函数来创建线程。

## 模块的常用方法
### `threading.active_count()`
判断当前线程数量

### `threading.current_thread()`
获取当前控制的线程对象

### `threading.get_ident()`
获取当前线程的线程标识符，线程销毁后，标识符可被复用。

### `threading.enumerate()`
获取当前线程对象列表

### `threading.main_thread()`
获取主线程对象

### `threading.stack_size([size])`
返回创建线程时的堆栈大小


## 通过传递函数来构造线程
```python
import threading
import time

def test(id=1):
    time.sleep(1)
    print("threading-%s start"%id)

th1 = threading.Thread(target=test)
th2 = threading.Thread(target=test, args=(2,))

th1.start()
th2.start()
print("all done")
```

可以注意到，当线程在等待1秒的期间，最后一句打印会被执行，因为程序不会阻塞。

## 通过继承*Thread*创建线程
```python
import threading
import time

class MyThread(threading.Thread):
    def __init__(self, id):
        self.id = id

    def run(self):
        time.sleep(1)
        print("threading-%s start"%id)

th1 = MyThread(1)
th2 = MyThread(2)

th1.start()
th2.start()
print("all done")
```

## 线程对象的常用属性及方法
### `start()`
开启线程活动，若调用该方法时，默认执行线程类中的`run()`方法。同一个线程对象只能调用一次。

### `join(timeout=None)`
让当前线程等待，直到线程结束。可以被调用多次。
```python
th = threading.Thread(target=func)
th.start()
th.join()
print("all done")
```

最后一局输出将会在线程结束后才调用。

### `is_alive()`
判断线程是否存活，返回布尔值。

### `ident`
获取线程的标识符

### `daemon`
判断该线程是否是守护线程，所有主线程创建的线程默认都为非守护线程

# `threading.Lock`
锁定当前线程，其他线程处于等待获取锁的状态。用于保护多线程下出现的不同线程获取到脏数据的情况。当锁被释放后，处于等待的线程可以获取锁。

锁不一定需要获取它的线程来释放。

## 常用方法
### `acquire(blocking=True, timeout=-1)`
获取锁，返回布尔值。`timeout=-1`表示无限阻塞等待。`blocking=True`表示`timeout`参数有效。

### `release()`
释放锁。

## 示例
```python
import threading
import time

lock = threading.Lock()

GLIST = [None] * 5

def test(param):
    lock.acquire()

    global GLIST
    for i in range(len(GLIST)):
        GLIST[i] = param
    
    print(GLIST)

    lock.release()

th1 = threading.Thread(target=test, args=('th1', ))
th2 = threading.Thread(target=test, args=('th2', ))

th1.start()
th2.start()
```

如果单个线程连续两次请求获取锁，将会造成死锁。

# `threading.RLock`
递归锁与普通锁的区别在于加入了*所属线程*和*递归等级*的概念。
- 所属线程：锁的释放必须由获取锁的线程来完成。
- 递归等级：当前线程再次获取锁时，锁的递归等级加1。初始等级为1

递归锁可以有效地解决死锁问题

## 示例
```python
import threading
import time

rlock_hi = rlock_hello = threading.RLock()

def test_thread_hi():
    rlock_hi.acquire()
    print("threading test_thread_hi got lock rlock_hi")
    time.sleep(2)
    rlock_hello.acquire()
    print("threading test_thread_hi got lock rlock_hello")
    rlock_hello.release()
    rlock_hi.release()

def test_thread_hello():
    rlock_hello.acquire()
    print("threading test_thread_hello got lock rlock_hello")
    rlock_hi.acquire()
    print("threading test_thread_hi got lock rlock_hi")
    rlock_hi.release()
    rlock_hello.release()

thread_hi = threading.Thread(target=test_thread_hi)
thread_hello = threading.Thread(target=test_thread_hello)
thread_hi.start()
thread_hello.start()
```

# `threading.Condition`
`Condition`允许一个或多个线程等待，直到被另一个线程通知。

## 常用方法
### `acquire(*args)`
请求锁

### `release()`
释放锁

### `wait(timeout=None)`
释放锁，阻塞线程，直到另一个线程使用`notify()`或`notify_all()`方法唤醒。被唤醒后会重新获取锁

### `wait_for(predicate, timeout=None)`
相当于：
```python
while not predicate:
    threading.Condition().wait()
```

该方法先调用`predicate`对象，如果`predicate`返回`False`，表示上面的`while`循环条件为`True`，执行`wait()`，释放锁并等待。

### `notify(n=1)`
唤醒一个正在等待的线程。`n=1`表示唤醒n个正在等待的线程。

该方法并不释放锁，所以当前线程如果调用了该方法，处于等待的线程也不能立刻获取锁，只有当前线程调用了`release()`方法释放了锁，等待的线程才能继续执行。

### `notify_all()`
唤醒所有等待的线程。

## 示例
```python
import threading
import time

condition_lock = threading.Condition()
PRE = 0

def pre():
    print(PRE)
    return PRE

def test_thread_hi():
    condition_lock.acquire()

    print("wait for threading test_thread_hello's notify")

    condition_lock.wait_for(pre)
    # 相当于：
    # while not pre:
    #     condition_lock.wait()
    
    # condition_lock.wait()
    print("continue execute")

    condition_lock.release()

def test_thread_hello():
    time.sleep(1)
    condition_lock.acquire()

    global PRE
    PRE = 1
    print("modify PRE to 1")
    print("notify thread test_thread_hello that it can prepare to get lock")
    condition_lock.notify()
    
    print("hello is ready to release lock")
    condition_lock.release()
    print("come and get lock")

thread_hi = threading.Thread(target=test_thread_hi)
thread_hello = threading.Thread(target=test_thread_hello)
thread_hi.start()
thread_hello.start()
```

# `threading.Semaphore`
信号量对象通过一个内部计数器管理同一时间可执行的最大线程数。获取锁时计数器减1，释放锁时计数器加一，当计数器为0时，阻塞线程，知道有线程通过释放锁使计数器增加，处于阻塞的线程才能获取锁。

## 示例
```python
import threading
import time

# 计数器初始值为3，表示同意之间只能有3个线程运行
semaphore3 = threading.Semaphore(3)

def thread_semaphore(index):
    semaphore3.acquire()
    time.sleep(2)
    print("thread_%s is running..." % index)
    semaphore3.release()

for index in range(9):
    threading.Thread(target=thread_semaphore, args={index,}).start()
```

# `threading.Event`
事件对象用于线程间分享状态，通过一个内部标识来实现。内部标识默认为`False`。

## 常用方法
### `is_set()`
返回内部标识

### `set()`
设置内部标识

### `clear()`
将内部标识设置为`False`

### `wait(timeout=None)`
阻塞线程，直到内部标识为`True`

## 示例
```python
import threading
import time

event = threading.Event()

def student_exam(id):
    print("student %s waiting for teacher to send papers" % id)
    event.wait()
    print("exam is begining")

def teacher():
    time.sleep(5)
    print("time up, begin to exam")
    event.set()

for id in range(3):
    threading.Thread(target=student_exam, args=(id,)).start()

threading.Thread(target=teacher).start()
```

# `threading.Barrier`
栅栏对象。阻塞调用了`wait()`方法的线程，直到所有目标线程都调用了`wait()`方法。然后同时释放所有线程。

```python
threading.Barrier(parties, action=None, timeout=None)
"""
parties: 指定线程数
action: 可调用对象。所有线程被释放前，随机抽取一个线程调用该方法
timeout: wait()的超时时间
"""
```

## 常用方法及属性
### `wait(timeout=None)`
阻塞线程，若超时，抛出`BrokenBarrierError`异常

### `reset()`
将栅栏对象重置为默认的初始态

### `abort()`
主动抛出`BrokenBarrierError`异常。

### `broken`
判断栅栏对象的的破损状态

### `n_waiting`
处于等待中的线程数量

### `parties`
通过栅栏的线程数量

## 示例
```python
import threading
import time

def test_action():
    print("call the func before all barrier threading released")

barrier = threading.Barrier(3, test_action)

def barrier_thread(sleep):
    time.sleep(sleep)
    print("barrier thread-%s wait" % sleep)
    barrier.wait()
    print("barrier thread-%s end" % sleep)

for sleep in range(6):
    threading.Thread(target=barrier_thread, args=(sleep,)).start()
```

# `threading.local`
## TLS
线程局部存储。允许线程访问同一个全局变量，每个线程会为自己局部存储一个副本，相互之间互不影响。

如果全局变量本身需要维护，那么需要加锁。

*flask*的上下文就基于此实现。

示例：
```python
import threading

userName = threading.local()

def SessionThread(userName_in):
    userName.val = userName_in
    print(userName.val)   

Session1 = threading.Thread(target=SessionThread("User1"))
Session2 = threading.Thread(target=SessionThread("User2"))

Session1.start()
Session2.start()

Session1.join()
Session2.join()
```

通过实现一个全局字典，*key*为线程ID，*value*为全局变量副本，当线程访问全局变量时，其实是根据线程ID得到了对应的副本。