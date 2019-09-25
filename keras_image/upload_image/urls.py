from django.conf import settings
from . import views
from django.conf.urls.static import static
from upload_image import views
from django.urls import path

app_name = 'upload_image'
urlpatterns = [
    path('', views.index, name="index"),
    path('loading/', views.upload, name='loading'),
    path('result/', views.result, name='result'),
]