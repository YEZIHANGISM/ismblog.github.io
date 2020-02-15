---
title: "rest_framework笔记"
categories:
  - Django
  - REST
tags:
  - rest_framework
---

# 认证与权限
## 认证
通过继承*BaseAuthentication*，并实现*authenticate*方法，可以实现自定义的认证函数
```
# authentication.py

from rest_framework import authentication

class MyAuthentication(authentication.BaseAuthentication):
    
    def authenticate(self, request):
        try:
            # programing your logic code here...
        except:
            raise exceptions.AuthenticationFailed("用户认证失败")

    # 认证失败，返回请求头信息
    def authenticate_header(self, request):
        pass
```

返回值：
- None:     该认证不做处理
- raise exception.AuthenticationFailed("用户认证失败")
- (user, obj)   (用户，模型对象自身)这两个变量将会赋值给request.user, request.auth

### 全局配置
在*settings.py*中配置*REST_FRAMWORK*

```
# settings.py

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [MyAuthentication,],
}
```

> 具体配置可参考**rest_framework.settings.api_settings.DEFAULTS**

### 局部配置
如果使用了全局配置，那么对于所有视图函数都会有效，对于不需要认证的函数，可以在视图中设置*authentication_classes*属性，为空表示该视图不做认证配置。

```
# views.py

from path.to.authentication import MyAuthentication

class CustomAPIView:
    authentication_classes = [MyAuthentication,]
```

## 权限
继承*BasePermission*，并实现*has_permission*方法，可以实现自定义的权限函数
```
# permissions.py

from rest_framework import permissions

class MyPermissions(permissions.BasePermission):

    def has_permission(self, request, view):
        # programing your logic code here...
```

返回值：
- True
- False

> BasePermission类中还有一个has_object_permission方法，当视图继承自ModelViewSet类时，实现该方法。

### 全局配置
```
# settings.py

REST_FRAMEWORK = {
  ...
  "DEFAULT_PERMISSION_CLASSES": [MyPermission, ],
}
```

### 局部配置
在视图函数中定义*permission_classes*属性
```
# views.py

from path.to.permission import MyPermission

class CustomAPIView:
    permission_classes = [MyPermission, ]
```

# 节流
控制访问频率

通过继承*BaseThrottle*，并实现*allow_request*和*wait*方法
```
# throttle.py

from rest_framework import throttling

class MyThrottle(throttling.BaseThrottle):
    
    def allow_request(self, request, view):
        # programing your logic code here...
        return True

    def wait(self):
        # 返回一个剩余等待的数字
        return 10
```

返回值
- True
- False

### 全局配置
在*settings.py*中配置
```
# settings.py

REST_FRAMEWORK = {
  "DEFAULT_THROTTLE_CLASSES": [MyThrottle, ]
}
```

### 局部配置
```
# views.py

from path.to.throttle import MyThrottle

class CustomAPIView:
    throttle_classes = [MyThrottle, ]
```