import base64
import datetime
import hashlib
import json
import logging
import os
import pickle
import random
import re
import string
import subprocess
import uuid
from dataclasses import dataclass
from hashlib import md5
from io import BytesIO
from random import randint
from xml.dom.pulldom import START_ELEMENT, parseString
from xml.sax import make_parser
from xml.sax.handler import feature_external_ges

import jwt
import requests
import yaml
from argon2 import PasswordHasher
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.core import serializers
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.template import loader
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageMath
from requests.structures import CaseInsensitiveDict

from .forms import NewUserForm
from .models import (FAANG, AF_admin, AF_session_id, Blogs, CF_user, authLogin,
                     comments, info, login, otp, sql_lab_table, tickits)
from .utility import customHash, filter_blog

#*****************************************Lab Requirements****************************************************#

#*****************************************Login and Registration****************************************************#

def register(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect('/')
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="registration/register.html", context={"register_form":form})

# def register(request):
#     if request.method=="POST":
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             form.save()
#         return redirect("login")

#     else:
#         form=UserCreationForm()
#         return render(request,"registration/register.html",{"form":form,})

def home(request):
    if request.user.is_authenticated:
        return render(request,'introduction/home.html',)
    else:
        return redirect('login')

## authentication check decurator function 
def authentication_decorator(func):
    def function(*args, **kwargs):
        if args[0].user.is_authenticated:
            return func(*args, **kwargs)
        else:
            return redirect('login')
    return function

#*****************************************XSS****************************************************#


def xss(request):
    if request.user.is_authenticated:
        return render(request,"Lab/XSS/xss.html")
    else:
        return redirect('login')

def xss_lab(request):
    if request.user.is_authenticated:
        q=request.GET.get('q','')
        f=FAANG.objects.filter(company=q)
        if f:
            args={"company":f[0].company,"ceo":f[0].info_set.all()[0].ceo,"about":f[0].info_set.all()[0].about}
            return render(request,'Lab/XSS/xss_lab.html',args)
        else:
            return render(request,'Lab/XSS/xss_lab.html', {'query': q})
    else:
        return redirect('login')
        

def xss_lab2(request):
    if request.user.is_authenticated:
        
        username = request.POST.get('username', '')
        if username:
            username = username.strip()
            username = username.replace("<script>", "").replace("</script>", "")
        else:
            username = "Guest"
        context = {
        'username': username
                }
        return render(request, 'Lab/XSS/xss_lab_2.html', context)
    else:
        return redirect('login')
    
def xss_lab3(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            username = request.POST.get('username', '')
            # Remove only alphanumeric characters (letters and digits)
            # This allows special characters like []()!+ for JSFuck-style payloads
            pattern = r'[a-zA-Z0-9]'
            result = re.sub(pattern, '', username)
            context = {'code':result}
            return render(request, 'Lab/XSS/xss_lab_3.html',context)
        else:
            return render(request, 'Lab/XSS/xss_lab_3.html')
            
    else:        
        return redirect('login')

#***********************************SQL****************************************************************#

def sql(request):
    if request.user.is_authenticated:

        return  render(request,'Lab/SQL/sql.html')
    else:
        return redirect('login')

def sql_lab(request):
    if request.user.is_authenticated:

        name=request.POST.get('name')

        password=request.POST.get('pass')

        if name:

            if login.objects.filter(user=name):

                sql_query = "SELECT * FROM introduction_login WHERE user='"+name+"' AND password='"+password+"'"
                print(sql_query)
                try:
                    print("\nin try\n")
                    val=login.objects.raw(sql_query)
                except:
                    print("\nin except\n")
                    return render(
                        request, 
                        'Lab/SQL/sql_lab.html',
                        {
                            "wrongpass":password,
                            "sql_error":sql_query
                        })

                if val:
                    user=val[0].user
                    return render(request, 'Lab/SQL/sql_lab.html',{"user1":user})
                else:
                    return render(
                        request, 
                        'Lab/SQL/sql_lab.html',
                        {
                            "wrongpass":password,
                            "sql_error":sql_query
                        })
            else:
                return render(request, 'Lab/SQL/sql_lab.html',{"no": "User not found"})
        else:
            return render(request, 'Lab/SQL/sql_lab.html')
    else:
        return redirect('login')

#***************** INSECURE DESERIALIZATION***************************************************************#

def insec_des(request):
    if request.user.is_authenticated:
        return  render(request,'Lab/insec_des/insec_des.html')
    else:
        return redirect('login')

@dataclass
class TestUser:
