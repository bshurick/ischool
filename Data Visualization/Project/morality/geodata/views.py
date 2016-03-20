import json 
from django.http import HttpResponse
from surveys.models import Survey
from itertools import groupby 

DPES11_DICT = {-1:'Refused'
                ,1: 'Strongly disagree'
                ,2: 'Disagree'
                ,3: 'Somewhat disagree'
                ,4: 'Neither agree nor disagree'
                ,5: 'Somewhat agree'
                ,6: 'Agree'
                ,7: 'Strongly agree'}

EPQ1_DICT = {-1:'Refused'
                ,1: 'Strongly agree'
                ,2: 'Agree'
                ,3: 'Somewhat agree'
                ,4: 'Neither agree nor disagree'             
                ,5: 'Somewhat disagree'
                ,6: 'Disagree'
                ,7: 'Strongly disagree'}

KEY = {
    'Strongly agree':'stronglyAgree'
   , 'Agree':'agree'
   , 'Somewhat agree':'somewhatAgree'
   , 'Neither agree nor disagree':'neitherAgreeNorDisagree'
   , 'Somewhat disagree':'somewhatDisagree'
   , 'Disagree':'disagree'
   , 'Strongly disagree':'stronglyDisagree'
   , 'Refused':'refused'
}

def geodata(request):
	question = request.GET.get('question')
	output = {}
	if question == '1':
		matches = sorted([
                        (m.state
                        , KEY[DPES11_DICT[int(float(m.i_develop_strong_emotions_toward_people_i_can_rely_on))]])
                        for m in Survey.objects.all()
                ])
		groups = [ (k[0],list(g)) for k, g in groupby(matches) ]
		for g in groups:
    			if g[0] not in output:
        			output[g[0]] = {}
    			for v in g[1]:
        			if v[1] not in output[g[0]]:
            				output[g[0]][v[1]] = 0 
        			output[g[0]][v[1]] += 1
	elif question == '2':
		matches = {}
	elif question == '3':
		matches = {}
	else: 
		raise Exception('Question needs to be 1-3')
	return HttpResponse(json.dumps(output),content_type='application/json')
