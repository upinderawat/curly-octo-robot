from django.shortcuts import render

def about(request):
    return render(request, 'covid19/about.html', {'title':'About'})

def login(request):
    return render(request, 'covid19/login.html', {'title':'Login'})