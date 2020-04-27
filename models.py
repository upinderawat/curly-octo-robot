from django.db import models

# Create your models here.

class Delivery_Agent(models.Model):
	id = models.AutoField(primary_key=True)
	name= models.CharField(max_length=100)
	password=models.CharField(max_length=50,default=0)
	phone_no=models.CharField(max_length=15)
	mode=models.CharField(max_length=10)
	location_lat=models.DecimalField(max_digits=9, decimal_places=6,default=0.0)
	location_lon=models.DecimalField(max_digits=9, decimal_places=6,default=0.0)

class Services(models.Model):
	id = models.IntegerField(primary_key=True,default=0)
	Name = models.CharField(max_length=100,default=0)
	serviceProviderType = models.IntegerField(default=1) 
	latitude = models.FloatField(default=0.0)
	longitude = models.FloatField(default=0.0)
	loggedIn = models.BooleanField(default=False)
	
   
class Orders(models.Model):
    orderId = models.IntegerField(primary_key=True)
    customerId = models.IntegerField()    
    fulfilled = models.BooleanField(default=False)
    latitude = models.FloatField(default=0.0)
    longitude = models.FloatField(default=0.0)
    # Order Type
    # 1. Groceries
    # 2. Ambulance
    # 3. Medical Supply
    # 4. Water Supply
    orderType = models.IntegerField(default=0)
     

class Delivery_Tracking(models.Model):
	del_id=	models.ForeignKey(Delivery_Agent, on_delete=models.CASCADE)
	service_id=models.ForeignKey(Services, on_delete=models.CASCADE)
	order_id=models.ForeignKey(Orders, on_delete=models.CASCADE)      	