import email
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail
from chatbot import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import authenticate, login, logout
from . tokens import generate_token
from django.shortcuts import render
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import nltk
import numpy as np
import torch
import json
import random
import os
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('chatbot/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = os.path.join(os.getcwd(), 'chatbot/NeuralNet.pth')
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "GMRIT"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I'm sorry, but I'm not sure what you mean by " + msg + " Can you please provide some additional context or clarify your question?"

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        text = data['message']
        response = get_response(text)
        message = {"answer": response}
        return JsonResponse(message)
    return JsonResponse({"error": "Invalid Request"})
@csrf_protect
def home(request):
    if request.session.get('email'):
        return render(request, 'index.html')
    else:
        return redirect('register')
@csrf_protect
def register(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')
        
        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')
        
        # if not username.isalnum():
        #     messages.error(request, "Username must be Alpha-Numeric!!")
        #     return redirect('home')
        

        myuser = User.objects.create_user(username, email, pass1)
        myuser.email = email
        myuser.first_name = fname
        myuser.last_name = lname
        # myuser.is_active = False
        myuser.is_active = False
        myuser.save()
        messages.success(request, "Your Account has been created succesfully!! Please check your email to confirm your email address in order to activate your account.")
        
        # Welcome Email
        subject = "Welcome to GMR HEALTH-CARE CHATBOT-Login!!"
        message = "Hello " + myuser.username + "!! \n" + "Welcome to GMR HEALTH-CARE CHATBOT!! \nThank you for visiting our website\n We have also sent you a confirmation email, please confirm your email address. \n\nThanking You\nSai Kiran"        
        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
        # Email Address Confirmation Email
        current_site = get_current_site(request)
        email_subject = "Confirm your GMR HEALTH-CARE CHATBOT-Login!!"
        message2 = render_to_string('email_confirmation.html',{
            
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser)
        })
        email = EmailMessage(
        email_subject,
        message2,
        settings.EMAIL_HOST_USER,
        [myuser.email],
        )
        email.fail_silently = True
        email.send()
        
        return redirect('login')
        
        
    return render(request, "register.html")

@csrf_protect
def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser, token):
        myuser.is_active = True
        # user.profile.signup_confirmation = True
        myuser.save()
        login(request, myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('login')
    else:
        return render(request, 'activation_failed.html')

@csrf_protect
def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        check_user = User.objects.filter(username=username, password=pass1)
        if check_user:
            request.session['user'] = username
            return redirect('home')
        user = authenticate(username=username, password=pass1)
        if user is not None:
            login(request, user)
            fname = user.first_name
            # messages.success(request, "Logged In Sucessfully!!")
            # return render(request, "index.html",{"fname":fname})
            request.session['myuser'] = username
            request.session['email'] =user.email
            request.session['username']=user.username
            return redirect('index')
        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('home')
    
    return render(request, "login.html")


def user_logout(request):
    del request.session['email']
    logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('home')



@csrf_protect
@login_required
def appoint(request):
    if request.method == "POST":
        gmail = request.POST['gmailid']
        subject = request.POST['subject']
        messages = request.POST['messages']
        user_email = request.user.email
        username = request.user.username
        # Welcome Email
        subject =  subject
        message = "from " +user_email  +" \n"  +gmail+" \n"  + messages      
        from_email = settings.EMAIL_HOST_USER
        to_list =  ['chitturidurgasatyasaikiran@gmail.com']
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
        # Welcome Email
        subject =  "Welcome to GMR HEALTH-CARE CHATBOT"
        message = "Hi " +username +" \n"  "Thanks For Your Feedback!!\n Thanks for spending your valuable time with our website."
        from_email = settings.EMAIL_HOST_USER
        to_list =  [user_email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        return render(request, 'sent.html')
    

@csrf_protect
@login_required
def index(request):
    if 'email' in request.session:
        return render(request, 'index.html')
    else:
        return redirect("login")