---
titile: "在ORM中实现主键从特定值开始自增"
categories:
  - Django
tags:
  - signals
  - django
  - python
---

# MYSQL
在`MySQL`中，可以通过指定`auto_increment`来设置自增的起始值
```SQL
CREATE TABLE TEST(
id int not null auto_increment primary key,
name varchar(127) not null default 'ISM'
)auto_increment=10000;
```

在`Django`中，可以使用`Signals`来设置主键从特定值开始自增。

首先是准备工作，创建一个`model`类
```python
class ISMModel(models.Model):
    name = models.CharField()
```

这个`model`将会创建一个默认的id字段作为主键，但是它的自增是从1开始的。所以我们可以在初始化时插入一条`id=9999`的数据然后删除，这样后续插入的数据就会从`id=10000`开始。

假设一个`django`项目结构:
```
project
    - project
        - settings.py
    - app1
        - __init__.py
        - models.py
        - apps.py
        ...
```

在`apps.py`中加入以下代码
```python
# apps.py

from django.apps import AppConfig
from django.db.models.signals import post_migrate


# 定义一个回调函数，在该函数中创建一条初始化数据然后删除
# 该函数必须接收一个sender参数
def increment_callback(sender, **kwargs):
    from app1.models import TestModel

    if sender.name == 'app1':
        try:
            test_model = TestModel.objects.create(id=9999, ...)
            test_model.delete()
        except:
            pass

class App1Config(AppConfig):
    name = 'app1'

    def ready(self):
        # 注册改回调函数
        post_migrate.connect(increment_callback, sender=self)
```

以上，我们使用了`post_migrate`来注册回调，表示当`migrate`执行后，发送该信号。

> 注意。导入模型类不能写到函数之外，因为此时app1尚未注册，会抛出注册错误
    ```
    django.core.exceptions.AppRegistryNotReady: Apps aren't loaded yet.
    ```

还有很多其他的信号可选，如`pre_migrate`，表示在`migrate`执行之前发送该信号。详细的信号可参考官方文档 [Signals](https://docs.djangoproject.com/en/2.2/ref/signals/#post-migrate)

在`settings.py`的`INSTALL_APPS`中加入`app1`
```python
# settings.py

...
INSTALL_APPS = [
    ...
    'app1.apps.App1Config'
]
```
> 注意，这里不能单纯的写作`app1`，因为我们的目的在于导入该app时触发信号。

以上，当执行`migrate`后，插入的数据ID将从10000开始递增。