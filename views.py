from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login
from django.template.loader import get_template
from django.http import HttpResponse, Http404
from django.db import models
# from django.contrib.auth.models 
from django.http import HttpResponse
from .models import Delivery_Agent 
from .models import Services
from .forms import PostForm 
from django.db.models import F

def index(request):
	all_del_agents=Delivery_Agent.objects.all()
	html='<h1>List Of Delivery Agents</h1>'
	for del_agent in all_del_agents:
		url='/delivery_login/'+str(del_agent.id)+'/'
		html+= '<a href= " ' + url +' ">' + del_agent.name+ '</a><br> ' 
	return HttpResponse(html)

def detail(request,del_id):
	obj=Delivery_Agent.objects.get(id=del_id)
	return HttpResponse("<h1>Details for delivery Agent with id " + str(obj.id)+ " : " + obj.name+" "+ str(obj.mode)+" at location : "+ str(obj.location_lat)+ ","+str(obj.location_lon) + "</h1>")


def selectDeliveryAgent(request,serviceprovider_id):
	sprovider = get_object_or_404(Services, pk=serviceprovider_id)
	alpha = 2
	alpha = alpha**2
	all_agents = Delivery_Agent.objects.all().filter(mode="free")
	all_agents = all_agents.annotate(result=(F('location_lat')-sprovider.latitude)**2+(F('location_lon')-sprovider.longitude)**2).filter(result__lt=alpha)
	context= {'allagents': all_agents}
	return render(request, 'delivery_login/del_list.html', context)

def login_view(request):

		 
	if request.method == 'POST':
 	  form=AuthenticationForm(data=request.POST)
 	  if form.is_valid():
 	  	# print("helooo")
 	  	return render(request,'delivery_login/del_home.html')

	
	else:
		# If the request is a GET request then, 
        # create an empty form object and  
        # render it into the page 
		# form = PostForm(None) 
		# print("helooo")
		form=AuthenticationForm()   
	return render(request, 'delivery_login/del_login.html',{'form':form}) 
	  # form=AuthenticationForm()
	  # return render(request,'delivery_login/del_login.html')  	

