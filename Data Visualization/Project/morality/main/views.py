from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

import random

def index(request):
	r = int(random.random()*10000)
	if r % 2 == 0:
		video = ''' 
			<iframe width="600" height="315" src="https://www.youtube.com/embed/2DN0IRoMf4k" frameborder="0" allowfullscreen></iframe> 
		'''
	else:
		video = ''' 
			<iframe width="600" height="315" src="https://www.youtube.com/embed/V6-0kYhqoRo" frameborder="0" allowfullscreen></iframe>
		'''
	tmp = loader.get_template('video.html')
	return HttpResponse(tmp.render({'video':video}, request))



