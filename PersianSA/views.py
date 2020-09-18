from django.shortcuts import render

# Create your views here.
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


@api_view(['GET'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([IsAuthenticated])
def get_user_files(request, user_id):
    # result = []
    # files = File.objects.filter(user=user_id)
    # file_serializer = FileSerializer(files, many=True, context={'request': request})
    #
    # for file in file_serializer.data:
    #     categories = Category.objects.filter(file=file['id'])
    #     category_serializer = CategorySerializer(categories, many=True)
    #     result.append({
    #         **file,
    #         'categories': category_serializer.data
    #     })

    return Response("doroste"+ str(user_id), status=status.HTTP_200_OK)
