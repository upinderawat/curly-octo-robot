from django.conf.urls import url
from . import views
# from django.contrib.auth.views import delivery_home

# app_name='delivery_login'

urlpatterns = [
url(r'^$', views.index,name='index'),
url(r'^login/$',views.login_view,name='login'),
url(r'^(?P<del_id>[0-9]+)/$',views.detail,name='detail'),
url(r'^selectDeliveryAgent/(?P<serviceprovider_id>[0-9]+)/(?P<order_id>[0-9]+)/$',views.selectDeliveryAgent,name='deliveryAgentDetails'),
url(r'^deliveryAgentAssigned/(?P<del_id>[0-9]+)/(?P<order_id>[0-9]+)/$',views.showAssignedAgent,name='assignedAgent')
# url(r'^delivery_home/$',views.index,name='index'))
# url(r'^$',views.login_view,name='login')
]