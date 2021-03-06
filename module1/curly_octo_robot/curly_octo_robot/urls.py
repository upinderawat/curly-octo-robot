"""curly_octo_robot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from covid19 import views as covid19_views
from user_accounts import views as user_accounts_views
from sp_accounts import views as sp_accounts_views
from delivery_accounts import views as delivery_accounts_views
from manage_orders import views as manage_orders_views
from django.urls import path

urlpatterns = [
    path('',covid19_views.login, name='start'),
    path('admin/', admin.site.urls),
    path('place_order', manage_orders_views.place_order_view, name='place_order'),
    path('about/', covid19_views.about, name='about'),
    path('login/', covid19_views.login, name='login'),
    path('register/',user_accounts_views.registration_view, name='user_register'),
    path('sp_register/',sp_accounts_views.registration_view, name='sp_register'),
    path('delivery_register/',delivery_accounts_views.registration_view, name='delivery_register')
]
