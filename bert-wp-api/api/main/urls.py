from unicodedata import name
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index,  name='index' ),
    path('ans', views.get_answer, name='answer')
]
