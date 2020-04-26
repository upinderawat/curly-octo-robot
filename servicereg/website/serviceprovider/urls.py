from django.conf.urls import url
from . import views

app_name = 'serviceprovider'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.login_user, name='login_user'),
    url(r'^logout/$', views.logout_user, name='logout_user'),
    # url(r'^ordersdetail/$', views.index, name='index'),
    url(r'^(?P<serviceprovider_id>[0-9]+)/$', views.serviceProviderDetails, name='sd'),
    url(r'^orderdetails/$', views.index, name='index'),
    url(r'^orderdetails/(?P<order_id>[0-9]+)/$', views.orderDetails, name='orderdetails'),
    url(r'^orderdetails/itemdetails/(?P<item_id>[0-9]+)/$', views.itemDetails, name='itemdetails'),
    url(r'^sporderdetails/(?P<serviceprovider_id>[0-9]+)/$', views.spOrderDetails, name='orderdetails'),
]
