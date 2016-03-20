import json 
from django.http import HttpResponse

def geodata(request):
	question = request.GET.get('question')
	if question == '1':
		matches = {}
	elif question == '2':
		matches = {}
	elif question == '3':
		matches = {}
	else: 
		raise Exception('Question needs to be 1-3')
	return HttpResponse(json.dumps(matches),content_type='application/json')
