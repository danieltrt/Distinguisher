from django.http import HttpResponse
from .src.logger import *
from .src.program import *
from .src.checker import *
from .src.interpreter import *
from .src.solver import *
from .src.model import *
from .src.distinguisher import *
from json import load, loads
from os import listdir
import sys
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.views.generic.edit import *
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from .models import *
from .forms import *
from django.shortcuts import redirect
from django.http import JsonResponse
import pickle
import time


logger = get_logger("dist")
logger.setLevel("DEBUG")
YESNO = "Yes/No"
OPTIONS = "Options"
f = open("./dist/example/instance1.json")
data = load(f)
f.close()


def yesno(request, choice_id=None, iter_n=None):
    answer = None
    if iter_n is None:
        selected_choice = get_object_or_404(Choice, pk=choice_id)
        dst = pickle.load(open(f'./distinguisher_{selected_choice.question_id}.bin', 'rb'))
        dst.update_programs(selected_choice.correctness)
        answer = selected_choice.correctness
    else:
        dst = pickle.load(open(f'./distinguisher_{iter_n}.bin', 'rb'))

    try:
        inpt, output = dst.distinguish()
    except ValueError as e:
        logger.error(str(e))
        return HttpResponseRedirect(reverse("dist:index"))

    if inpt is True and output is True:
        return render(request, 'success.html', {'program': dst.get_answer(answer)})
    question = Question(id=None, question_text=inpt, interaction_model=YESNO)
    question.save()
    for out in output:
        choice = Choice(id=None, question_text=question, choice_text=out, question_id=question.id)
        choice.save()
    pickle.dump(dst, open(f'./distinguisher_{question.id}.bin', 'wb'))
    return render(request, 'yesno.html', {
        'question': question, "header": inpt.get_header(), "rows": inpt.get_active_rows()
    })


def options(request, choice_id=None, iter_n=None):
    answer = None
    if iter_n is None:
        selected_choice = get_object_or_404(Choice, pk=choice_id)
        dst = pickle.load(open(f'./distinguisher_{selected_choice.question_id}.bin', 'rb'))
        dst.update_programs(selected_choice.choice_text)
        answer = selected_choice.choice_text
    else:
        dst = pickle.load(open(f'./distinguisher_{iter_n}.bin', 'rb'))

    try:
        inpt, output = dst.distinguish()
    except ValueError as e:
        logger.error(str(e))
        return HttpResponseRedirect(reverse("dist:index"))

    if inpt is True and output is True:
        return render(request, 'success.html', {'program': dst.get_answer(answer)})
    question = Question(id=None, question_text=inpt, interaction_model=OPTIONS)
    question.save()
    for out in output:
        choice = Choice(id=None, question_text=question, choice_text=out, question_id=question.id)
        choice.save()
    pickle.dump(dst, open(f'./distinguisher_{question.id}.bin', 'wb'))
    return render(request, 'options.html', {
        'question': question, "header": inpt.get_header(), "rows": inpt.get_active_rows()
    })


def submit(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if question.interaction_model == OPTIONS:
        key = "choice"
        selected_choice = question.choice_set.get(pk=request.POST[key])
        selected_choice.save()
        return HttpResponseRedirect('/unchartit' + reverse('dist:options_opt', kwargs={'choice_id': selected_choice.id}))

    elif question.interaction_model == YESNO:
        correctness = None

        if "choice_yes" in request.POST:
            key = "choice_yes"
            correctness = True
        elif "choice_no" in request.POST:
            key = "choice_no"
            correctness = False

        selected_choice = question.choice_set.get(pk=request.POST[key])
        selected_choice.correctness = correctness
        selected_choice.save()

        return HttpResponseRedirect('/unchartit' + reverse('dist:yesno_opt', kwargs={'choice_id': selected_choice.id}))


def index(request):
    return render(request, 'distinguisher.html')


def upload(request):
    logger.debug("Loading CBMC template.")

    # build constraints list
    json = loads(request.POST['inputConstraints'])
    input_constraints = json_to_cprover(json)
    constraints = [input_constraints, int(request.POST['nRows']),
                   int(request.POST['nCols']) + 1, 8, 24, list(json.keys()) + ['NEW'],
                   [json[key][0] for key in json.keys()]]
    template = UnchartItTemplate(data['cbmc_template'], constraints)
    interpreter = UnchartItInterpreter(constraints)
    programs = []
    col_mapping = {list(json.keys())[i]: i for i in range(len(json.keys()))}
    for file_name in list(request.FILES.keys()):
        programs += [UnchartItProgram(f=request.FILES[file_name], n_cols=int(request.POST['nCols']), vars=col_mapping)]

    # Generic
    model_checker = CBMC(template)
    solver = Solver("open-wbo")

    if request.POST['interactionModel'] == YESNO:
        interaction_model = YesNoInteractionModel(model_checker, solver, interpreter)
        dst = Distinguisher(interaction_model, programs, int(time.time()))
        pickle.dump(dst, open(f'./distinguisher_{dst.n}.bin', 'wb'))
        return HttpResponse(reverse("dist:yesno_iter", args=(0, dst.n,)))

    elif request.POST['interactionModel'] == OPTIONS:
        interaction_model = OptionsInteractionModel(model_checker, solver, interpreter)
        dst = Distinguisher(interaction_model, programs, int(time.time()))
        pickle.dump(dst, open(f'./distinguisher_{dst.n}.bin', 'wb'))
        return HttpResponse(reverse("dist:options_iter", args=(0, dst.n,)))


