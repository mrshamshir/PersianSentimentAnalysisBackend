from django.urls import path
from .views import *

urlpatterns = [

    path('result', get_result),

]
