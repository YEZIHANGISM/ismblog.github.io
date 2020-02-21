---
title: "rest_framework笔记"
categories:
  - Django
  - REST
tags:
  - rest_framework
---

# 认证
通过继承*BaseAuthentication*，并实现*authenticate*方法，可以实现自定义的认证函数
```python
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

## 全局配置
在*settings.py*中配置*REST_FRAMWORK*

```python
# settings.py

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [MyAuthentication,],
}
```

> 具体配置可参考**rest_framework.settings.api_settings.DEFAULTS**

## 局部配置
如果使用了全局配置，那么对于所有视图函数都会有效，对于不需要认证的函数，可以在视图中设置*authentication_classes*属性，为空表示该视图不做认证配置。

```python
# views.py

from path.to.authentication import MyAuthentication

class CustomAPIView:
    authentication_classes = [MyAuthentication,]
```

# 权限
继承*BasePermission*，并实现*has_permission*方法，可以实现自定义的权限函数
```python
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

## 全局配置
```python
# settings.py

REST_FRAMEWORK = {
  ...
  "DEFAULT_PERMISSION_CLASSES": [MyPermission, ],
}
```

## 局部配置
在视图函数中定义*permission_classes*属性
```python
# views.py

from path.to.permission import MyPermission

class CustomAPIView:
    permission_classes = [MyPermission, ]
```

# 节流
控制访问频率

通过继承*BaseThrottle*，并实现*allow_request*和*wait*方法
```python
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

## 全局配置
在*settings.py*中配置
```python
# settings.py

REST_FRAMEWORK = {
  "DEFAULT_THROTTLE_CLASSES": [MyThrottle, ]
}
```

## 局部配置
```python
# views.py

from path.to.throttle import MyThrottle

class CustomAPIView:
    throttle_classes = [MyThrottle, ]
```

# 序列化器
序列化器允许*queryset*和模型实例等复杂的数据类型转化为python数据类型，并且能够简单地被渲染。
## Serializer
定义一个简单模型
```python
# models.py

class CustomModel(models.Model):
    email = models.EmailField()
    content = models.CharField()
    created = models.DateTimeField()
```

为*django*的每个*app*新建*serializers.py*
```python
# serializers.py

from rest_framework import serializers

class CustomSerializer(serializers.Serializer):
    email = serializers.EmailField()
    content = serializers.CharField()
    created = serializers.DateTimeField()
```
为数据库插入一些测试数据后，在*shell*中测试
```python
>>> from app.models import CustomModel
>>> from app.serializers import CustomSerializer
>>> custom = CustomModel.objects.all()
>>> serializer = CustomSerializer(instance=custom, many=True)
>>> serializer.data
# [OrderedDict([('email', 'test@example.com'), ('content', 'hello world'), ('created', '2020-02-18T22:52:13.398851Z')])]
```

### 字段
*serializer.data*中展示的字段，便是*CustomSerializer*中定义的字段，所以*CustomSerializer*中定义的字段必须在模型中存在。

对于一些特殊的字段，例如*ForeignKey*，可以添加*source*参数指定关联的模型。这样可以取到外键字段关联的字符串值，而非单纯的对象本身
```python
owner = serializers.CharField(source="owner.username")
```

源码中对*source*的处理方式是：如果*owner.username*是一个可调用对象，那么将返回*owner.username*()的结果，否则直接返回*owner.username*，所以*source*也可以填方法名，比如*get_absolute_url*

同理，对于*ManyToManyField*字段，可以使用*field.all*，但是这样的展示，只是将关联的模型对象展示出来，如果想要详细到字段的显示，可以通过自定义显示的方法。

### SerializerMethodField
*SerializerMethodField*字段可以自定义数据的展示，需要实现一个*get_fieldname*方法
```python
many = serializers.SerializerMethodField()

def get_many(self, row):
    '''
    函数名以get_开头，以字段名结尾
    '''

    results = row.many.all()
    
    ret = []
    for item in results:
        ret.append({"id":item.id, "name":item.name})

    return ret
    # return [
        {"id":1, "name":"karfka"},
        {...}
    ]
```

> 关于字段的扩展参数，可以参考官网[Serializer Fields](https://www.django-rest-framework.org/api-guide/fields/)

## ModelSerializer
自定义的序列化器类还可以继承*ModelSerializer*，相对有几个优点：
- 自动生成模型中存在的字段
- 自动为序列化器生成验证器
- 包含了简单的.create()和.update()的实现

```python
# serializers.py

from rest_framework import serializers

class CustomSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomModel
        fields = "__all__"
        read_only_fields = ["email"]    # 某个字段只读
        exclude = ["id"]    # 排除某个字段
        depth = 1   # 嵌套序列化，可以得到模型的关系型字段（比如ForeignKey）关联的其他模型的信息，深度为1，建议0-10
```

在*shell*中导入类，打印对象的相关信息
```python
>>> from app.serializers import CustomSerializer
>>> custom = CustomSerializer()
>>> custom
CustomSerializer():
    id = IntegerField(label='ID', read_only=True)
    email = EmailField(max_length=254)
    content = CharField(max_length=200)
    created = DateTimeField(required=False)
```

*ModelSerializer*类依然可以定义模型中不存在的字段，或者覆盖原有字段，或者自定义展示的字段。任何关系型字段，例如*ForeignKeyField*将会默认映射到*PrimaryKeyRelatedField*

## HyperlinkedModelSerializer
*HyperlinkedModelSerializer*使用超链接来表示关系，而非*ModelSerializer*使用的*PrimaryKeyRelatedField*。

默认情况下，序列化器将包含一个*url*字段。或者，你可以使用模型中的*ForrignKey*字段，在*serializer*类中使用*HyperlinkedIdentityField*重写该字段。这个字段将会直接返回一个拼接好的可访问的*url*，前提是你已经定义好该*url*的路由映射和视图函数。
```python
# serializers.py

from rest_framework import serializers
from app.models import CustomModel

class CustomSerializer(serializers.HyperlinkedModelSerializer):
    url = serializers.HyperlinkedIdentityField(
            view_name="detail",
            lookup_field='pk',
        )

    class Meta:
        model = CustomModel
        fields = ["url", "id", "email", "content"]
```

*view_name*表示*url*映射中对应的*name*。*lookup_field*表示通过*pk*关键字去寻找对应的实例。

当定义了*HyperlinkedIdentityField*，那么在实例化*CustomSerializer*时，必须将*request*添加到对象的上下文中去，确保生成完整的*url*。
```python
serializer = CustomSerializer(isntance=model, context={'request': request})
```

## 验证
发送POST请求时，可以在保存数据之前先对数据进行验证。
```python
serializer = CustomSerializer(data=data)
if serializer.is_valid():
    return serializer.validated_data
else:
    return serializer.errors
```

*serializers.errors*的错误信息可以定制
```python
# serializers.py

class CustomSerializer(serializers.Serializer):
    content = serializers.CharField(error_message={"required": "内容不能为空"})
```

### 字段级的验证
通过在*CustomSerializer*中实现*validate_fieldname*方法，可以对单独的字段进行验证，类似于*Django*的*form*表单中的*clean_fieldname*方法
```python
# serializers.py

class CustomSerializer(serializers.Serializer):
    password = serializers.CharField()

    def validate_password(self, value):
        if value != "*****":
            raise serializers.ValidationError("password didn't match")
        return value
```

### 对象级的验证
如果验证需要多个字段，可以在*CustomSerializer*类中实现*validate*方法
```python
# serializers.py

class CustomSerializer(serializers.Serializer):
    password = serializers.CharField()
    passwordrepeat = serializers.CharField()

    def validate(self, data):
        """
        :type data: dict
        """
        if data["password"] != data["passwordrepeat"]:
            raise serializers.ValidationError("password inconsistent")
        return data
```

### Validators
可以为单个字段添加多个自定义验证，验证也可重用
```python
class CustomValidation:

    def __init__(self, title):
        self.title = title

    def __call__(self, value):
        if not self.value.startswith(self.title):
            message = "title must start with %s" % self.value
            raise serializers.ValidationError(message)
```

接下来在字段中引用
```python
# serializers.py

class CustomSerializer(serializers.ModelSerializer):
    title = serializers.CharField(validators=[CustomValidation("ism_")])
```

# 通用视图
## APIView
*APIView*继承自*Django*的*View*，在此基础上增加了很多属性。
```python
class APIView(View):

    renderer_classes = api_settings.DEFAULT_RENDERER_CLASSES
    parser_classes = api_settings.DEFAULT_PARSER_CLASSES
    authentication_classes = api_settings.DEFAULT_AUTHENTICATION_CLASSES
    throttle_classes = api_settings.DEFAULT_THROTTLE_CLASSES
    permission_classes = api_settings.DEFAULT_PERMISSION_CLASSES
    content_negotiation_class = api_settings.DEFAULT_CONTENT_NEGOTIATION_CLASS
    metadata_class = api_settings.DEFAULT_METADATA_CLASS
    versioning_class = api_settings.DEFAULT_VERSIONING_CLASS
```

## GenericAPIView
*GenericAPIView*继承自*APIView*，是其他视图的通用基本视图。同样定义了一些属性以及常用方法
```python
class GenericAPIView(APIView):
    # 该属性不应该在方法中直接使用，而是调用`get_queryset()`获取
    queryset = None
    
    serializer_class = None

    lookup_field = 'pk'
    lookup_url_kwargs = None

    filter_backends = api_settings.DEFAULT_FILTER_BACKENDS
    
    pagination_class = api_settings.DEFAULT_PAGINATION_CLASS

    # 常用方法
    def get_queryset(self):...

    def get_object(self):...

    def get_serializer(self, *args, **kwargs):...

    ...
```

## mixins
*mixins*中定义了用于基本视图中的行为方法，即是常见的对*queryset*的*CURD*操作。*mixins*一共提供了五种行为操作的类。
- ListModelMixin：提供list方法，返回列表类型的结果集
- CreateModelMixin：提供create方法，创建和保存新的模型实例
- RetrieveModelMixin：提供retrieve方法，返回一个详细的模型实例信息
- UpdateModelMixin：提供update方法，修改和保存已有模型实例
- DestoryModelmixin：提供delete方法，删除模型实例

以下是一些通过继承*mixins*提供的一个或多个类以及通用视图*GenericAPIView*得到的视图函数。
- CreateAPIView
- ListAPIView
- RetrieveAPIView
- DestroyAPIView
- UpdateAPIView
- ListCreateAPIView
- RetrieveUpdateAPIView
- RetrieveDestroyAPIView
- RetrieveUpdateDestroyAPIView

# ViewSet（视图集）
*ViewSet*指将一系列相关视图组合在一个单个类中，形成一套处理逻辑。*ViewSet*不提供处理函数如*get*或*post*，只提供动作函数如*list*或*create*。
```python
# views.py

from rest_framework import viewsets

class CustomViewSet(viewsets.ViewSet):
    def list(self, request):
        # return list of queryset

    def retrieve(self, request, pk=None):
        # return detail of one model instance
```

*ViewSet*类继承自*ViewSetMixin*和*APIView*，*ViewSetMixin*中对*as_view*方法做了修改，接收*action*参数，根据参数来分发给目标动作函数。
```python
custom_list = CustomViewSet.as_view(action={"get":"list"})
custom_detail = CustomViewSet.as_view(action={"get":"retrieve"})
```

> 注意：通常情况下，我们不会这样做，而是向路由系统注册ViewSet，让url自动生成。这会在**路由**一节中介绍

常用的视图集类：
- ViewSet
- GenericViewSet
- ModelViewSet
- ReadOnlyModelViewSet

## action装饰器
如果你有特别的方法也需要路由，那么可以为其添加*action*装饰器
```python
# views.py

class CustomViewSet(ViewSet):

    @action(detail=False, methods=["get"])
    def another_queryset(self, request):
        # return queryset
```

`detail=False`表示该方法返回一个结果集，不然就是返回一个对象。

然后同样为其分发一个动作函数
```python
another = CustomViewSet.as_view(action={"get":"another_queryset"})
```

# 路由

# 版本
版本控制允许在不同的客户端定制不同的行为。

## 全局配置
```python
# settings.py

REST_FRAMEWORK = {
    "DEFAULT_VERSIONING_CLASS": "rest_framework.versioning.QueryParameterVersioning"
    
    # Optional
    "DEFAULT_VERSION": None
    "ALLOWED_VERSIONS": None
    "VERSION_PARAM": "version"
}
```

*rest_framework*提供几种常用的版本控制方式，源码中有例子
- QueryParameterVersioning：通过url中的参数控制版本
- AcceptHeaderVersioning：通过将版本添加到请求头的方式
- URLPathVersioning：通过url路径
- NamespaceVersioning：通过django的namespace
- HostNameVersioning：子域名

## 局部配置
通过在视图中定义versioning_class属性。
```python
# views.py

class CustomView(APIView):
    versioning_class = versioning.URLPathVersioning
```

> 注意：单独设置并不推荐，全局的版本控制应该是单一的。

# 解析器
解析器允许请求附带各种数据类型，如*json*、*form*等。解析器会检查传入的请求的*Content-Type*头，然后使用对应的解析器解析，内存将保存在*request.data*属性中。

## 全局配置
```python
# settings.py

REST_FRAMEWORK = {
    "DEFAULT_PARSER_CLASSES": [
        'rest_framework.parsers.JSONParser',
    ]
}
```

> 更多解析类请阅读源码：rest_framework.parsers

## 局部配置
```python
# views.py

from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView

class CustomView(APIView):
    parser_classes = [JSONParser, ]

    def post(self, request, *args, **kwargs):
        return Response({"code":"200", "data":request.data})
```

# 分页
## 全局配置
```python
# settings.py

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 100
}
```

> 注意：上面两项配置都是必须的

## 局部配置
通过在视图函数中定义*pagination_class*属性。
```python
# views.py

class CustomView(APIView):
    pagination_class = PageNumberPagination
```
## 默认的三种分页风格
- PageNumberPagination
基于页码的分页风格。例如：页码为1，数据为10条。
- LimitOffsetPagination
基于位置与限量的分页风格。例如：从第10条的位置向后取10条数据。
- CursorPagination
基于指针的分页风格，只显示上一页和下一页。

## 自定义的分页类
通过实现定制的分页类，可以定制分页的属性，同时也能继承分页的功能
```python
from rest_framework import pagination

class CustomPagination(pagination.PageNumberPagination):
    page_size = 100

    # url中请求页码的参数
    page_query_param = 'page'

    # url中亲贵每页结果集数量的参数
    page_size_query_param = "number"

    def paginate_queryset(self, queryset, request, view):
        ...
```

> 更多的属性配置请参考官方文档：[Pagination](https://www.django-rest-framework.org/api-guide/pagination/)

### 在视图中使用
```python
# views.py

from app.models import CustomModel
from path.to.serializers import CustomSerializer
from path.to.pagination import CustomPagination
from rest_framework.response import Response

class CustomView(APIView):
    def get(self, request, *args, **kwargs):
        custom = CustomModel.objects.all()
        
        page = CustomPagination()
        page_result = page.paginate_queryset(queryset=custom, request=request, view=self)

        serializer = CustomSerializer(instance=page_result, many=True)
paginate_queryset

        return Response(serializer.data)
```