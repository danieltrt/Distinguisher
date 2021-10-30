from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.urls import reverse
import ast
from django.shortcuts import render, get_object_or_404

# Create your views here.


def index(request):
    return render(request, 'synthesizer.html')


def upload(request):
    return HttpResponseRedirect('/unchartit' + reverse('synth:programs', args=(1,)))


def programs(request, id):
    return render(request, 'programs.html', {
        "programs": ['df0 <- input %>% filter(a > 0)', 'df0 <- input %>% filter(a >= 1)']
    })

