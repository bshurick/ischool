from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from .models import Survey
from .forms import SurveyForm

def survey(request):
	if request.method == 'POST':
		f = SurveyForm(request.POST)
		if f.is_valid():
			f.save()
	else:
		f = SurveyForm()
	tmp = loader.get_template('survey_form.html')
	return HttpResponse(tmp.render({'form':f}, request))


