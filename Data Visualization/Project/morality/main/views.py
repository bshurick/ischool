from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

import random, requests
from geodata.views import get_geodata, get_fillkey

def index(request):
	videos = {
		'test':'''
                                <iframe width="600" height="315" src="https://www.youtube.com/embed/2DN0IRoMf4k" frameborder="0" allowfullscreen></iframe>
                        '''
		,'control':'''
                                <iframe width="600" height="315" src="https://www.youtube.com/embed/V6-0kYhqoRo" frameborder="0" allowfullscreen></iframe>
                        '''

	}
	if not request.session.get('video'):
		r = int(random.random()*10000)
		if r % 2 == 0:
			video = videos['control']
			request.session['video'] = 'control'
		else:
			video = videos['test']
			request.session['video'] = 'test'
	else:
		video = videos[request.session.get('video')]
	tmp = loader.get_template('index.html')
	watched_video = request.session.get('watched_video',False)
	c = {
		'video':video
		,'watched_video':watched_video
		,'fillkey':get_fillkey()
		,'map1data':get_geodata(1)
		,'map2data':get_geodata(2)
		,'map3data':get_geodata(3)
	}
	return HttpResponse(tmp.render(c,request))



