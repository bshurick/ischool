
from django.conf.urls import url
from main.views import index

app_name='main'
urlpatterns = [
    url(r'^', index, name='index')
]

