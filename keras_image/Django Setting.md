# Django Setting

## 1. 가상환경

```bash
# 파이썬 버전 확인
# 반드시 3.7.x 버전이 맞는지 확인 후 진행
$ python -V
Python 3.7.4


# 가상환경 생성
# python -m venv <가상환경 설치경로>
$ python -m venv venv


# 가상환경 적용
$ source venv/Scripts/activate


# 버전확인
(venv)
$ python -V
Python 3.7.4


# 설치된 모듈 확인
(venv)
$ pip list
Package    Version
---------- -------
pip        19.0.3
setuptools 40.8.0


# pip upgrade
(venv)
$ python -m pip install --upgrade pip


# pip upgrade 확인
(venv)
$ pip list
Package    Version
---------- -------
pip        19.2.2  # <- upgrade version
setuptools 40.8.0
```





---

## 2. VS Code 및 기타세팅

### VS Code 파이썬 환경 세팅

- VS Code Extensions 에서 `Python` 과 `Django` 설치
- `Ctrl + Shift + P ` => `python select interpreter` => 방금 생성한 가상환경을 선택 (.\venv\Scripots\python.exe)
  - .vscode/settings.json 파일이 생성되며 터미널에서 자동으로 가상환경 적용된다면 OK



### Git ignore 세팅

- `gitignore.io`에 접속해서 `python`, `django`, `windows`, `vscode` 선택 후 생성
- `.gitignore` 파일 생성 후 붙여넣기



### VS Code Django 환경 세팅

```json
{
    // 파이썬 환경 선택 => 자동으로 해줌
    "python.pythonPath": "venv\\Scripts\\python.exe",

    // Django 에서 사용되는 파일 타입에 대한 정의
    "files.associations": {
        "**/templates/*.html": "django-html",
        "**/templates/*": "django-txt",
        "**/requirements{/**,*}.{txt,in}": "pip-requirements"
    },

    // django-html 에서도 html emmet을 적용
    "emmet.includeLanguages": {"django-html": "html"},

    // django-html 에서 tab size를 2칸으로 고정
    "[django-html]": {
        "editor.tabSize": 2
    },
}
```





---

## 3. Start Django Project

```bash
(venv)
$ pip install django
```

- Django를 설치한 순간부터 `django-admin`이라는 command를 사용할 수 있게 된다.
- 이 command를 통해 django project에 여러가지 명령을 할 수 있다.



### Start Project

```bash
(venv)
$ django-admin startproject django_intro .
```

- 현재 디렉토리에서 django_intro라는 이름으로 프로젝트를 시작하겠다,.

- Django project naming

  - -캐릭터는 사용될 수 없다.

  - python과 django에서 이미 사용되는 이름은 사용하지 않는다.

    (django라는 이름은 django 그 자체와 충돌이 발생하며, test라는 이름도 django 내부적으로 사용하는 모듈이름)



### Run server

```bash
$ python manage.py runserver
```

- ctrl + c 로 종료
- 기본적으로 `localhost:8000`에서 실행된다.

