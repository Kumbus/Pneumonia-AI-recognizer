from django.http import JsonResponse
from .models import Image
from .serializers import ImageSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import random

def is_pneumonia(image_path):
    image_path = image_path.replace("/", "\\")
    image_path = image_path[1:]
    with open(image_path, 'r') as f:
        if random.randint(0,1) == 0:
            return {'isPneumonia': False}
        else:
            return {'isPneumonia': True}
        
@api_view(['GET','POST'])
def images_list(request):

    if request.method == 'GET':
        images = Image.objects.all()
        serializer  = ImageSerializer(images, many=True)
        return JsonResponse({'images': serializer.data})
    
    if request.method == 'POST':
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            #ai working
            hasIllness = is_pneumonia(serializer.data.get('image'))
            return Response({'isPneumonia': hasIllness}, status=status.HTTP_201_CREATED)
        




