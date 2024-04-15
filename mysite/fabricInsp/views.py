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
        if len(defectTiles) > 0:
            addContours(img, defectTiles)

            # is_success, buffer = cv2.imencode(".png", img)
            # pngTile = io.BytesIO(buffer)
            # image64 = image_to_base64(pngTile)

            image64 = cv_to_base64(img)

            args = {}
            args['image64'] = image64
            args['show'] = True
            args['tiles'] = defectTiles

            return render(request, 'test.html', args)
        else:
            return render(request, 'test.html')
    else:
        return render(request, 'test.html')