from django.http import JsonResponse
from .models import Image
from .serializers import ImageSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import random
import time
import numpy as np
import tensorflow as tf
import cv2

def is_pneumonia(image_path):
    model = tf.keras.models.load_model('my_model.h5')
    image_path = image_path.replace("/", "\\")
    image_path = image_path[1:]

    img = cv2.imread('chest_xray/test/NORMAL/IM-0029-0001.jpeg')
    img_resized = cv2.resize(img, (180, 180))

    img_array = np.array(img_resized, dtype=np.float32)

    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model(img_batch)
    print(prediction)

        
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
            time.sleep(2)
            #ai working
            hasIllness = is_pneumonia(serializer.data.get('image'))
            return Response({'isPneumonia': hasIllness}, status=status.HTTP_201_CREATED)
        




