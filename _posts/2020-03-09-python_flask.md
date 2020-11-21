---
title: "flask"
categories:
  - flask
tags:
  - flask
---

# DEBUG模式
开启*DEBUG*模式有助于调试程序、代码自动重载以及更好的错误信息等服务。

## 开启方法
- `app.run(debug=True)`
- `app.debug = True`
- `app.config.update(DEBUG=True)`
- 在linux环境下将debug参数写入环境变量：`export FLASK_DEBUG=1`
- 导入自定义的配置文件，通过`app.config.from_objects('configs')`
- 在*pycharm*中右键运行按钮，设置*Edit Configuration*

# 路由
有两种方式为视图函数添加路由
1. 装饰器
2. `add_url_rule()`

*endpoint*类似*Django*的*urlpattern*中的url别名。
```python
app = Flask(__name__)
def mylist():
	return "list"

app.add_url_rule("/list/", endpoint="list", view_func=mylist)
```

`add_url_rule()`通常与类视图配合使用。

## url_for
通过视图函数构建对应的url

```python
from flask import Flask, url_for

@app.route("/index/<int:id>", endpoint="myindex")
def index(id):
	# url中未定义的参数将会作为url中的参数拼接
	# 生成的url为：/index/1?next=hello/
	return url_for("myindex", id=1, next="hello")
```

`url_for`的第一个参数是自定义的`endpoint`，如果没有定义，使用函数名。

## BaseConverter
通过继承`BaseConverter`可以实现自定义的url参数格式
```python
from flask import Flask
from werkzeug.routing import BaseConverter

class TelConverter(BaseConverter):
	"""
	实现一个匹配电话号码的url参数
	"""
	regex = r'1[85374]\d{9}'

app = Flask(__name__)
app.url_map.converters['tel'] = TelConverter
```

### `to_python(self, value)`
格式转换类中重载该方法，其返回值会传递给视图函数的参数

例如，在url中用`+`号分割：`www.reddit.com/r/flask+lisp/`，以同时访问两个社区的帖子
```python
from flask improt Flask
from werkzeug.routing import BaseConverter

class ListConverter(BaseConverter):
	def to_python(self, value):
		ret = value.split("+")
		return ret

app = Flask(__name__)
app.url_map.converters["list"] = ListConverter

@app.route("/r/<list:boards>")
def posts(boards):
	return boards
```

### `to_url(self, value)`
与`to_python`相反，在调用`url_for`时，该方法将返回值拼接到url中返回

在`ListConverter`中重载该方法
```python
from flask import Flask
from flask import url_for
from werkzeug.routing import BaseConverter

class ListConverter(BaseConverter):
	...

	def to_url(self, value):
		ret = "+".join(value)
		return ret

app = Flask(__name__)

@app.route("/d/")
def add():
	return url_for("posts", boards=["a", "b"])
```

## 重定向
```python
from flask import Flask
from flask import redirect
from flask improt request


app = Flask(__name__)

@app.route("/login/", methods=["POST",])
def login():
	return "login"

@app.route("/index/", methods=["GET", "POST"])
def index():
	name = request.args.get("name")

	if not name:
		return redirect(url_for("login/"))
	else:
		return name
```

# 模板
## 自定义过滤器
```python
from flask import Flask
from flask import render_template


app = Flask(__name__)

# 装饰器的参数可选，如果不提供，默认使用函数名作为过滤器名
@app.template_filter("reverse")
def reverse_filter(value):
	return value[::-1]

app.jinja_env.filters["reverse"] = reverse_filter
```

在模板中使用
```jinja
<h1>{{ name|reverse }}</h1>
```

## 宏
与`python`，将常用的代码片段抽取出来定义为宏，方便重用。
```
<!-- macro.html -->

{% macro input(name, type="text", value="") %}
	<input type="{{ type }}", name="{{ name }}" value="{{ value|e }}">
{% endmacro %}


```

在模板中引用
```jinja
{% from "macro.html" import input %}

<p>{{ input("password", type="password", value="enter pwd") }}</p>
```

### 传递上下文变量
如果需要渲染的模板中有需要传递到宏模板的变量。在导入模板的时候可以使用`with`关键字导入上下文变量
```jinja
<!-- index.html -->
{% from 'macro.html' import input with context %}

<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title>Document</title>
</head>
<body>
	<p>{{ username }}</p>
</body>
</html>
```

这样就可以在宏模板中使用`username`变量
```jinja
<!-- macro.html -->

{% macro input(name, type="text", value="") %}
	<input type="{{ type }}", name="{{ name }}" value="{{ username }}">
{% endmacro %}
```

## `set`与`with`
使用`set`在模板中定义变量，可供模板全局使用。
```jinja
{% set username="ismblog" %}
<p>{{ username }}</p>
```

使用`with`同样可以定义变量，在被包裹的代码块中使用
```jinja
{% with username="ism" %}
	<p>{{ username }}</p>
{% endwith %}
```

# 类视图
## View
```python
from flask import Flask
from flask import views

app = Flask(__name__)

class IndexView(views.VIew):
	def dispatch_request(self):
		return "list"

app.add_url_rule("/list/", endpoint="mylist", view_func=IndexView.as_view("list"))
```

## MethodView
*MethodView*类似*Django*的*View*，通过实现不同的方法，允许不同的请求方式。
```python
from flask import Flask
from flask import views
from flask import render_template
from flask import requet

app = Flask(__name__)

class LoginView(views.MethodView):
	def get(self):
		return render_template("login.html")

	def post(self):
		username = request.form.get("username")
		password = request.form.get("password")
		if username=="admin" and password=="admin":
			return "登录成功"
		else:
			error = "wrong password or username"
			return render_template("login.html", error=error)

```

## 类视图中使用装饰器
```python
class IndexView(views.View):
	decorators = [customdecor, ]
	...
```

# Blueprint
蓝图类似于*Django*的各个app，将所有视图按照功能分为各个模块，使项目结构更加清晰。

例如，将用户相关的视图都放到`users.py`中
```python
# users.py

from flask import Blueprint

# template_folder的相对路径根据import_name决定
bp_users = Blueprint("user", import_name=__name__, url_prefix="/users/", template_folder="users")

@bp_users.route("/profile")
def users_list():
	return "users"
```

在主模块中引用
```python
# app.py

from flask import Flask
import users

app = Flask(__name__)
app.register_blueprint(bp_users)
```

## 蓝图的模块查找顺序
优先从项目根目录下的*templates*目录中查找，如果没有，再到*template_folder*定义的路径下寻找

## 在蓝图中使用上下文
如果特定的函数需要使用到上下文，可以在函数内部导入app，避免外部导入带来的循环导入问题
```python
def foo():
	from app import create_app
	app = create_app
	with app.app_context():
		dosomething()
```

# flask-sqlalchemy
```python
pip install flask-sqlalchemy
```

使用flask-sqlalchemy与数据库进行交互，使用ORM来操作表。

flask-sqlalchemy基于SQLAlchemy，对其进行封装，使用更加方便。

## 基本使用
```python
# app.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import config

app = Flask(__name__)
app.config.from_object(config)

db = SQLAlchemy(app)

if __name__ == "__main__":
	app.run()
```

数据库配置
```python
# config.py

DB_URI = "sqlite:///db.sqlite3"
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = True
```

这里使用sqlite作为交互的数据库。

定义数据库表模型
```python
# models.py

from app import db

class User(db.Model):
	__tablename__ = "user"
	id = db.Column(db.Integer, primary_key=True, autoincrement=True)
	username = db.Column(db.String(32), nullable=False)
```

接下来可以在*shell*中测试数据库连接
```bash
>>> import app
>>> from models import User
>>> app.db.create_all()
>>> user = User(username="test")
>>> app.db.session.add(user)
>>> app.db.session.commit()
```

## 循环引用
以上代码在*shell*中测试没有问题，但是如果尝试在*app.py*中导入models后再进行测试，会抛出导入异常的错误
```bash
>>> import app
ImportError: cannot import name 'db' from 'app' 
```

这是因为*models.py*和*app.py*发生了循环引用。*app.py*导入了models，*models.py*导入了app。解决这个问题，我们可以定义一个新的文件，将db对象的实例化放到新的文件中
```python
# exts.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
```

修改app.py
```python
# app.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from exts import db
import config

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)

if __name__ == "__main__":
	app.run()
```

models.py
```python
from exts import db

...
```

# flask-script
```bash
pip install flask-script
```

flask-script支持自定义命令，并在命令行中使用。可以在命令行执行相应的程序。比如
- 创建管理员用户
- 与flask-migrate组合使用迁移数据库
- 更改启动flask服务的地址

## 基本使用
```python
# manage.py

from flask_script import Manager
from app import app

manager = Manager(app)

@manager.command
def greet():
    print("no args decorator")

@manager.option("-u", "--username", dest="name")
@manager.option("-p", "--password", dest="password")
def login(name, password):
    print("username: %s, password: %s" % (name, password))


if __name__ == "__main__":
    manager.run()
```

在*shell*中测试
```shell
$ python manage.py greet
$ python manage.py login -u admin -p admin
```

# flask-migrate
flask-migrate通常和flask-script配合使用，通过命令行完成数据库迁移

## 基本使用
```python
# manage.py

from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app import app
from exts import db

manager = Manager(app)
Migrate(app, db)
manager.add_command("db", MigrateCommand)

if __name__ == "__main__":
    manager.run()
```

> 假设你已定义好数据库表模型

数据库迁移初始化
```shell
python manage.py db init
```

这会在当前目录下生成一个*migration*目录，所有的更改都会保存在此目录下。

> 注意，运行以上命令时有可能会报一个错误: `No changes in schema detected.`
这是因为我们将模型定义在了`models.py`中，但是整个程序周期中并没有导入models，所以找不到定义的表。

在`app.py`或者`models.py`中导入models即可
```python
# app.py

import models
...
```

接下来跟踪数据库模型
```shell
$ python manage.py db migrate
```

这条命令将会扫描SQLAlchemy对象，将所有的更改记录到一个py文件中，并分配一个版本号。

将更改映射到数据库
```shell
$ python manage.py db upgrade
```

版本回退，首先获取历史版本号，再回滚到某一版本
```python
$ python manage.py db history
$ python manage.py db downupgrade <id>
```

# flask-WTF
```shell
pip install flask-WTF
```

flask-WTF提供了对*WTForms*的集成，功能类似于*Django*的*Form*表单。用于表单验证、渲染模板、文件管理以及CSRF保护等。
```python
# forms.py

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

class MyForm(Form):
	name = StringField("name", validators=[DataRequired()])
```

> 注意：`flask_wtf.Form更名为FlaskForm，后期会被删除，所以建议使用FlaskForm`

> `flask_wtf.FlaskForm`与`wtforms.Form`的区别在于：
> 1. 前者加入了csrf的验证。
> 2. 如果使用`wtforms.Form`，那么在实例化表单类时，需要添加`request.form`参数：`form=MyForm(reuqest.form)`，使用`FlaskForm`则不需要。

在模板中定义一个表单：
```
<!-- index.html -->

<form action="" method="post">
    {{ form.csrf_token }}
    <table>
        <tr>
            <td>用户名：</td>
            <td><input type="text" name="name"></td>
        </tr>
        <tr>
            <td></td>
            <td><input type="submit" value="提交"></td>
        </tr>
    </table>
</form>
```

在视图中处理验证请求：
```python
# app.py
from flask import Flask, request, render_template
from forms import MyForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "dfhjsajkhfadsl"

@app.route("/submit/", methods=["GET", "POST"])
def submit():
	form = MyForm()
	if requet.method == "GET":
		return render_template("index.html")
	else:
		if form.validate():
			return "success"
		else:
			print(form.errors)
			return "fail"
```

## 常用验证器
- Email：验证数据是否为邮箱
- EqualTo(fieldname)：验证数据是否和参数字段的数据一致
- InputRequied：验证是否输入数据
- Length(min, max)：长度限制
- NumberRange(min, max)：数字范围限制
- Regexp：自定义正则表达式
- URL
- UUID

> 可参考wtforms.validators模块

## 自定义验证器
与*Django*类似，在表单类中自定义验证方法
```python
# forms.py

from wtforms.validators import ValidationError

class MyForm(Form):
	name = StringField("name", validators=[DataRequired()])

	def validate_name(self, field):
		if field.data != "admin":
			raise ValidationError("wrong name")
```

抛出的异常信息将会被`form.errors`捕获。

## 上传文件
定义表单
```python
# forms.py

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

class PhotoForm(FlaskForm):
	photo = FileField(validators=[FileRequired()])
```

模板
```
<!-- upload.html -->

<form action="" method="post" enctype="multipart/form-data">
    {{ form.csrf_token }}
    <table>
        <tbody>
            <td>
                <tr>上传文件：</tr>
                <tr><input type="file" name="photo"></tr>
            </td>
            <td>
                <tr><input type="submit" value="上传"></tr>
            </td>
        </tbody>
    </table>
</form>
```

注意表单的enctype属性一定要设置，否则无法识别提交类型。

视图函数：
```python
# app.py

from werkzeug.utils import secure_filename
import forms
import os

STATIC_PATH = os.path.join(os.path.dirname(__file__), "static")

@app.route("/upload/", methods=("GET", "POST"))
def upload():
    form = forms.PhotoForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(STATIC_PATH, 'images', filename))
    else:
    	return redirect(url_for("index"))
    return render_template("upload.html", form=form)
```

上传的文件保存在`request.files`中。

> 上传的文件名最好做一层安全的转换，因为文件上传到服务器是有风险的。

### 文件验证
验证文件是否为服务器希望的文件类型，避免上传恶意文件或上传错误的文件类型。

文件验证可以与flask-Uploads配合使用，也可以单独使用flask-WTF完成。

```python
# forms.py

from flask_wtf.file import FileAllowed

class PhotoForm(FlaskForm):
	photo = FileField(validators=[
			FileRequired(),
			FileAllowed(["png",], messages=".png only")
		])
```

和flask-Uploads使用
```python
# forms.py

from flask_wtf.file import FileAllowed
from flask_uploads import UploadSet, IMAGES

images = UploadSet("images", IMAGES)

class PhotoForm(FlaskForm):
	photo = FileField(validators=[
			FileRequired(),
			FileAllowed(images, messages=".png only")
		])
```

视图函数：
```python
@app.route("/upload/", methods=("GET", "POST"))
def upload():
    form = forms.PhotoForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(STATIC_PATH, 'images', filename))
    else:
        filename = None
    return render_template("upload.html", form=form, filename=filename, errors=form.errors)
```

在模板中添加错误信息
```
<form action="" method="post" enctype="multipart/form-data">
    {{ form.csrf_token }}
    <table>
        <tbody>
            <td>
                <tr>上传文件：</tr>
                <tr><input type="file" name="photo"></tr>
            </td>
            <td>
                <tr><input type="submit" value="上传"></tr>
            </td>
        </tbody>
    </table>
    {% if errors %}
        <p>{{ errors }}</p>
    {% endif %}
</form>
```

## CSRF保护
为flask设置全局的scrf保护
```python
# app.py

from flask_wtf.csrf import SCRFProtect

csrf = CSRFProtect(app)
```

为form表单提供csrf
```
<form action="" method="post">
	{{ form.csrf_token }}
</form>
```

在ajax请求中使用csrf_token
```jquery
var csrftoken = $('input[name="csrf_token"]').val();
$.post({
	...
	"data": {
		...,
		"csrf_token": csrftoken
	}
	})
```

# 请求生命周期
1. 运行flask时，执行`app.run()`，*run*又调用了*werkzeug.serving*的*run_simple*方法，这个方法在内部创建套接字，监听端口，等待请求。

2. 当有用户请求时，执行`app.__call__()`，接着调用*wsgi_app*方法。

3. 在*wsgi_app*中，首先执行`ctx = self.request_context(environ)`，返回一个*RequestContext*对象，里面包含了*request*和*session*两个重要的属性。然后执行`ctx.push()`将*ctx*入栈。

4. 在*push*中，首先检查本地栈对象中是否非空，如果是，则将其出栈，然后执行`app_ctx = self.app.app_context()`，获得应用上下文，依次将应用上下文和请求上下文入栈，最后给*session*赋值。

5. 分发请求。并在此之前先执行*process_request*，这个方法执行所有被*before_request*装饰的函数，然后通过路由分发执行视图函数。最后执行*finalize_request*

6. 在*finalize_request*中，执行`response = self.make_response(rv)`，将视图函数的返回值封装到*response*中，执行`response = self.process_response(response)`。

7. 在*process_response*中，执行所有被*after_request*装饰的函数，保存*session*，然后返回响应。

8. 最后，将栈中的对象销毁。

flask中，每一个请求的用户都会拥有自己的request对象，并在请求中全局使用。*request*只是动态的全局变量，只在flask的请求周期中存在。

## 应用上下文
### flask.current_app
指向请求的当前应用实例。

### flask.g
即*global*，替代*python*的全局变量用法，保存全局数据。

## 请求上下文
### flask.request
当前请求的request对象。

### flask.session
当前请求的session对象。

# 钩子函数
在flask中，钩子函数是由装饰器装饰的函数，在执行视图函数前先执行钩子函数

- before_first_request - 第一次请求时执行，之后都不再执行。
- before_request - 每一次请求到视图函数前先执行。
- teardown_appcontext - 每一次请求到视图函数后执行。
- template_filter - 自定义过滤器。
- context_processor - 传递上下文，以在模板中使用。返回值必须是字典。
- errorhandler - 接收状态码，定义该状态码的处理方式。

# 信号
```shell
pip install blinker
```

## 基本使用
```python
import blinker

def foo(sender):
	print(sender)
	print("this is a signal")

# 创建信号对象
appsignal = blinker.Namespace()
login_signal = appsignal.signal("login_signal")

# 监听信号
login_signal.connect("foo")

# 发送信号
appsignal.send()
```

> 常用内置信号可参考`flask.signals`

# `flask-RESTful`
```python
pip install flask-restful
```

基本使用
```python
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Helloworld(Resource):
	def get(self):
		return {"hello": "world"}

api.add_resource(Helloworld, "/", endpoint="hello")

if __name__ == "__main__":
	app.run()
```

路由中也可以指定多URL，所有URL都会指向视图类：
```python
api.add_resource(Helloworld, "/", "/hello/", endpoint="hello")
```

## 参数解析
flask-RESTful内置了对表单数据的验证，用法类似于python的`argparse`模块。
```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument("username", type=string, help="wrong username")
args = parser.parse_args()	# return type: python dict
```

`type`可以解析一些内置的类型，比如`int`， `string`等，也可以使用*flask_restful.inputs*来验证一些其他的类型
- inputs.url - 验证url
- inputs.date - 验证日期
- inputs.regex - 验证正则表达式
- inputs.boolean - 验证布尔值
- inputs.positive - 验证正整数
- inputs.natural - 验证自然数

*add_argument*常用参数
- default - 默认值
- required - 是否必须
- choices - 参数的可选项容器
- trim - 去除数据的空格
- nullable - 是否可以为空

## 数据格式化
支持将一个对象序列化输出
```python
from flask_restful import fields, marshal_with

resource_fields = {
	'task': fields.String,
	'url': fields.Url("hello")
}

class Foo(object):
	def __init__(self, id, task):
		self.id = id
		self.task = task

		# 这个字段不会发送到响应中
		self.status = "active"

foo = Foo(id="1", task="remember")

class CustomView(Resource):
	@marshal_with(resource_fields)
	def get(self, **kwargs):
		return foo
```

- fields.List - 返回一个列表
- fields.Nested - 嵌套字典

### *fields*参数
- default - 默认值
- attribute - 重命名字段

## 渲染HTML模板
```python
api.representation("text/html")
def output_html(data, code, headers):
	print(data)
	resp = make_response(data)
	return resp

class CustomView(Resouce):
	def get(self):
		return render_template("custom.html")
```

# `flask-Mail`
```shell
pip install flask-Mail
```

```python
# app.py

from flask import Flask
from flask_mail import Mail, Message
import os

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.qq.com'  # 邮件服务器地址
app.config['MAIL_PORT'] = 25               # 邮件服务器端口
app.config['MAIL_USE_TLS'] = True          # 启用 TLS
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME') or 'me@example.com'
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD') or '123456'

mail = Mail(app)

@app.route('/')
def index():
    msg = Message('Hi', sender='me@example.com', recipients=['he@example.com'])
    msg.html = '<b>Hello Web</b>'
    # msg.body = 'The first email!'
    mail.send(msg)
    return '<h1>OK!</h1>'

if __name__ == '__main__':
    app.run()
```

也可以把配置写入配置文件
```python
# config.py

MAIL_SERVER = "smtp.qq.com"
MAIL_PORT = 465  # 非加密协议端口号为25
MAIL_USE_TLS = False    # 端口号587
MAIL_USE_SSL = True    # 端口号465
MAIL_USERNAME = "xxxx@qq.com"
MAIL_PASSWORD = "xxxxxx"
MAIL_DEFAULT_SENDER = "xxxx@qq.com"
```

## 异步发邮件

## 发送附件邮件