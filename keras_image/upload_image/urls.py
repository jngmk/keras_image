from django.conf import settings
from . import views
from django.conf.urls.static import static
from upload_image import views
from upload_image.views import CreateImageView
from django.urls import path

urlpatterns = [
    path('', CreateImageView.as_view()),
    path('loading/', views.upload),
    path('result/', views.result),
]