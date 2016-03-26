from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader

from .models import Survey
from .forms import SurveyForm

def survey(request):
	if request.method == 'POST':
		video = request.session.get('video')
		postdata = request.POST.copy()
		postdata['video_choice'] = video
		f = SurveyForm(postdata)
		if f.is_valid():
			f.save()
			return HttpResponseRedirect('/')
	else:
		f = SurveyForm()
	tmp = loader.get_template('survey_form.html')
	return HttpResponse(tmp.render({'form':f}, request))


