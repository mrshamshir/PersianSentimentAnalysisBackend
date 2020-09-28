from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from Analyzer.analyzer import get_classes


@api_view(['POST'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([IsAuthenticated])
def get_result(request):

    comment = request.data["comment"]
    classifier = request.data["classifier"]
    test_size = request.data["percentage"]

    table, result = get_classes(comment, classifier, test_size)
    # result = get_classes(result,classifier ,test_size )

    return Response(table, status=status.HTTP_200_OK)
