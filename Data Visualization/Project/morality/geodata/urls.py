
from django.conf.urls import url
from geodata.views import geodata

urlpatterns = [
    url(r'^', geodata),
]

