from django.contrib import admin
from .models import Orders, Items, Service, Order_Dispatch

admin.site.register(Orders)
admin.site.register(Items)
# admin.site.register(Services)
admin.site.register(Service)
admin.site.register(Order_Dispatch)