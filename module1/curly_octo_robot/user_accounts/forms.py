from django import forms
from django.contrib.auth.forms import UserCreationForm

from user_accounts.models import Account

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(max_length=60)

    class Meta:
        model = Account
        fields = ('email','username','password1','password2','latitude','longitude')