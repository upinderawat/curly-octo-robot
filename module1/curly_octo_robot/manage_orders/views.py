from django.shortcuts import render,redirect
from django.contrib.auth import login,authenticate
from manage_orders.forms import OrderForm

def place_order_view(request):
    context = {}
    if request.POST:
        form = OrderForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/login')
        else:
            context['place_order'] = form
    else:
        form = OrderForm()
        context['place_order'] = form
    return render(request, 'manage_orders/order.html', context)