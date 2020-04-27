from django import forms

from manage_orders.models import Order

class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        exclude = ('customer_id',)