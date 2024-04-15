from django.http import HttpResponse
from django.shortcuts import render
from .inspection import *
import base64



model = loadModel()

cnn_isp = CnnInspector(6, 25, model, 5, 5, 15)

def index(request):
    return render(request, "test.html", {})


def upload_image(request):
    global cnn_isp
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        cnn_isp.setImg(image)
        defectTiles, tiles, predictions, probailities, clusters, img = cnn_isp.inspect()

        args = {}

        if len(defectTiles) > 0:
            addContours(img, defectTiles)
            image64 = cv_to_base64(img)

            args['image64'] = image64
            args['tiles'] = defectTiles
            args['lable'] = 'Defects Found'

            return render(request, 'tester.html', args)
        else:
            image64 = cv_to_base64(img)
            args['image64'] = image64
            args['lable'] = 'No Defects Found'
            return render(request, 'tester.html', args)
    else:
        return render(request, 'tester.html')