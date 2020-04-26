from django.db import models
from django.urls import reverse

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
    def get_absolute_url(self):
        return reverse('serviceprovider:orderdetails', kwargs={'pk':self.pk})
    def __str__(self):
        s=str(self.orderId) + ' - ' + str(self.customerId) + ' - ' + str(self.fulfilled)+'-'
        s+=str(self.latitude) + ' - ' + str(self.longitude) + ' - ' +  str(self.orderType)
        return s

class Items(models.Model):
    itemId = models.IntegerField(primary_key=True)
    oId = models.ForeignKey(Orders, on_delete=models.CASCADE)
    provider_id = models.IntegerField()   
    # fulfilled = models.BooleanField(default=False)

    def __str__(self):
        return str(self.itemId) + ' - ' + str(self.oId) + ' - ' + str(self.provider_id)

class Service(models.Model):
    serviceProviderId = models.IntegerField(primary_key=True)
    serviceProviderName = models.CharField(max_length=250, default="Groceries")
    # Service Provider Type
    # 1. Groceries
    # 2. Ambulance
    # 3. Medical Supply
    # 4. Water Supply
    serviceProviderType = models.IntegerField(default=1)    
    latitude = models.FloatField(default=0.0)
    longitude = models.FloatField(default=0.0)
    loggedIn = models.BooleanField(default=False)
    def __str__(self):
        s = str(self.serviceProviderName) + ' - ' + str(self.serviceProviderType) + ' - '
        s+= str(self.latitude) + ' - ' + str(self.longitude) + ' - ' + str(self.loggedIn)
        return s

class Order_Dispatch(models.Model):
	service_id=models.ForeignKey(Service, on_delete=models.CASCADE)
	order_id=models.ForeignKey(Orders, on_delete=models.CASCADE)