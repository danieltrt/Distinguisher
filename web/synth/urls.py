from django.urls import path
from . import views

app_name = 'synth'
urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('programs/<int:id>/', views.programs, name='programs')
]


