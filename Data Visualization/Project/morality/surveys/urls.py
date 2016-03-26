
from django.conf.urls import url
from surveys.views import survey

app_name = 'surveys'
urlpatterns = [
    url(r'^', survey, name='survey'),
]

