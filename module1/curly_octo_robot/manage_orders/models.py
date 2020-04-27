from django.db import models

# Create your models here.
class Order(models.Model):
    order_id= models.CharField(max_length=256)
    customer_id = models.CharField(max_length=256)
    list_of_items= models.TextField(max_length=500)