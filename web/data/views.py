from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.urls import reverse
from .models import Chart
import ast
from django.shortcuts import render, get_object_or_404


# Create your views here.

def index(request):
    return render(request, 'data.html')


def upload(request):
    from .src.predict import chart_to_table
    res = None
    for file_name in list(request.FILES.keys()):
        f = request.FILES[file_name]
        res = chart_to_table(f, int(request.POST['min']), int(request.POST['max']))
        chart = Chart(id=None, file_name=file_name)
        chart.save()
        with open(f'extracted_csv_{chart.id}.csv', 'a+') as f:
            f.write(str(res))
        return HttpResponse(reverse('data:display', args=(chart.id,)))


def display(request, id):
    chart = get_object_or_404(Chart, pk=id)
    with open(f'extracted_csv_{chart.id}.csv', 'r+') as f:
        res = ast.literal_eval(f.read())
    return render(request, 'data_display.html', {
        "header": ['COL0', 'COL1'], "rows": [[f'bar{i}', f'{res[1][i]:.4f}'] for i in range(res[0])]
    })
