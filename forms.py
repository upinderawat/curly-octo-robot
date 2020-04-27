from django.forms import ModelForm 
from django import forms 
from delivery_login.models import Delivery_Agent

# define the class of a form 
# from formValidationApp.models import * 
class PostForm(ModelForm): 
	class Meta: 
		# write the name of models for which the form is made 
		model = Delivery_Agent		 

		# Custom fields 
		fields =["id","name", "password","phone_no","mode","location_lat","location_lon"] 

	# this function will be used for the validation 
	def clean(self): 

		# data from the form is fetched using super function 
		super(PostForm, self).clean() 
		
		# extract the username and text field from the data 
		username = self.cleaned_data.get('name') 
		password = self.cleaned_data.get('password') 

		# conditions to be met for the username length 
		if len(username) < 5: 
			self._errors['username'] = self.error_class([ 
				'Minimum 5 characters required'])
		obj=Delivery_Agent.objects.all().filter(name=username)
		if(obj.password	 != password):
		# if len(text) <10: 
			self._errors['password'] = self.error_class([ 
				'pasword wrong']) 

		# return any errors if found 
		return self.cleaned_data 
