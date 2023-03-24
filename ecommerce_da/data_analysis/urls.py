from django.urls import path
from . import views
app_name = 'data_analysis'
urlpatterns = [
 # Home page.
 path('', views.index, name='index'),
]