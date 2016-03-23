
from django.conf.urls import url
from surveys.views import survey

urlpatterns = [
    url(r'^', survey),
]

