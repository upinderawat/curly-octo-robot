from django.contrib import admin
from .models import Delivery_Agent
from .models import Services
from .models import Orders
from .models import Delivery_Tracking
# Register your models here.

admin.site.register(Delivery_Agent)
admin.site.register(Services)
admin.site.register(Orders)
admin.site.register(Delivery_Tracking)