from django.urls import path
from .views import *



urlpatterns = [

    path('result/<int:user_id>', get_user_files),

]

