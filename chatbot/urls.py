from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register', views.register, name='register'),
    path('activate/<uidb64>/<token>', views.activate, name='activate'),
    path('login',views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
    path('predict', views.predict, name='predict'),
    path('appoint', views.appoint, name='appoint'),
    path('index/', views.index, name='index'),

]


