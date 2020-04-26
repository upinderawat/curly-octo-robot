from django.shortcuts import render
from django.http import HttpResponse
import datetime
from .models import Orders, Items, Service, Order_Dispatch
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from .forms import UserForm
from django.db.models import Q
from django.template import loader
from django.db.models import F, Func


def index(request):
    now = datetime.datetime.now()

    # html = "<html><body>It is now %s.</body></html>" % now
    # # return HttpResponse(html)
    all_orders = Orders.objects.all()
    # template = loader.get_template('')
    context= {'allorders': all_orders}
    return render(request, 'serviceprovider/orderlist.html', context)
    # html='<html><body>'
    # for order in all_orders:
    #     print(str(order.orderId))
    #     url = '/serviceprovider/'+str(order.orderId)+'/'
    #     html+= '<a href-"'+url+'">'+order.orderId+'</a><br>'
    # html+='</body></html>'
    # html = "<html><body>It is now %s.</body></html>" % now
    # return HttpResponse(html)
    # return render(request, 'serviceprovider/test.html')

def orderDetails(request, order_id):
    # html = "<html><body>"+ str(order_id)+"</body></html>"
    # album = get_object_or_404(Orders, pk=order_id)
    # return render(request, 'serviceprovider/t2.html')
    # return render(request, 'serviceprovider/t2.html', {'album': album})
    # return HttpResponse("<h2>Details for album id "+str(order_id)+"</h2>")
    order = get_object_or_404(Orders, pk=order_id)
    items = Items.objects.all().filter(oId=order_id)
    return render(request, 'serviceprovider/orderdetails.html', {'order': order, 'allitems':items})

def itemDetails(request, item_id):
    items = get_object_or_404(Items, pk=item_id)
    return render(request, 'serviceprovider/itemdetails.html', {'items': items})

def serviceProviderDetails(request, serviceprovider_id):
    sprovider = get_object_or_404(Service, pk=serviceprovider_id)
    return render(request, 'serviceprovider/providerdetails.html', {'sprovider': sprovider})

def spOrderDetails(request, serviceprovider_id):
    sprovider = get_object_or_404(Service, pk=serviceprovider_id)
    alpha = 2
    alpha = alpha**2
    all_orders = Orders.objects.all().filter(orderType=sprovider.serviceProviderType, fulfilled=False)
    all_orders = all_orders.annotate(result=(F('latitude')-sprovider.latitude)**2+(F('longitude')-sprovider.longitude)**2).filter(result__lt=alpha)
    context= {'allorders': all_orders}
    return render(request, 'serviceprovider/odlist.html', context)

def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                # all_orders = Orders.objects.all()
                # context= {'allorders': all_orders}
                # return render(request, 'serviceprovider/orderlist.html', context)
                print(user.username)
                sprovider = get_object_or_404(Service, serviceProviderName=user.username)
                return render(request, 'serviceprovider/providerdetails.html', {'sprovider': sprovider})
                # return render(request, 'serviceprovider/serviceprovider/orderlist', {'albums': all_order})
            else:
                return render(request, 'serviceprovider/login.html', {'error_message': 'Your account has been disabled'})
        else:
            return render(request, 'serviceprovider/login.html', {'error_message': 'Invalid login'})
    return render(request, 'serviceprovider/login.html')


def register(request):
    form = UserForm(request.POST or None)
    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user.set_password(password)
        user.save()
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                # albums = Album.objects.filter(user=request.user)
                # return render(request, 'serviceprovider/index.html', {'albums': albums})
    context = {
        "form": form,
    }
    return render(request, 'serviceprovider/registration.html', context)

def logout_user(request):
    logout(request)
    form = UserForm(request.POST or None)
    context = {
        "form": form,
    }
    return render(request, 'serviceprovider/login.html', context)
