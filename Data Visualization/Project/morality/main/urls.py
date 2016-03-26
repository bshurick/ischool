
from django.conf.urls import url
from main.views import index

urlpatterns = [
    url(r'^', index)
]

