import fastbook
from fastbook import *
from fastai.vision.widgets import *
import cv2
import numpy as np

from io import BytesIO
from scipy import misc
import base64


def cv_to_base64(image):
    is_success, buffer = cv2.imencode(".png", image)
    pngTile = io.BytesIO(buffer)
    img_str = base64.b64encode(pngTile.getvalue())
    img_str = img_str.decode("utf-8") 
    return img_str


class Tile:   
    x = 0
    y = 0
    width = 0
    height = 0
    suspects = None
    roi = None
    prob = None
    defect = False
    coded = None

    def __init__(self,x, y, width, height, suspect):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.suspects = []
        self.suspects.append(suspect)

    def setDefect(self, prob):
        self.prob = float(prob)
        self.defect = True
        self.coded = cv_to_base64(self.roi)
    

class Inspector():
    K = None
    opening = None
    closing = None
    blur = None
    img = None
    c_size = None


    def __init__(self, K, c_size, opening, closing, blur):
        self.K = K
        self.c_size = c_size 
        self.opening = opening
        self.closing = closing
        self.blur = blur

    def setParams(self, K, c_size, opening, closing, blur):
        self.K = K
        self.c_size = c_size 
        self.opening = opening
        self.closing = closing
        self.blur = blur

    def setImg(self, img):
        frame = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
        temp = frame.real
        self.img = temp

    def checkParams(self):
        if self.K == None: return False
        if self.opening == None: return False
        if self.closing == None: return False
        if self.closing == None: return False
        return True

    def cluster(self):

        img = cv2.bilateralFilter(self.img,self.blur,75,75)

        Z = img.reshape((-1,3)) 
        Z = np.float32(Z) 
    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    
        ret,label,center=cv2.kmeans(Z,self.K,None,criteria,attempts = 10, flags = cv2.KMEANS_RANDOM_CENTERS) 
        center = np.uint8(center) 
        res = center[label.flatten()] 
        res2 = res.reshape((img.shape)) 
        
        center_n = []

        for i in center:
            center_n.append(i[0])

        empty = np.zeros((img.shape[0], img.shape[1]))
        empty = np.uint8(empty) 

        custerContainer = []
        finalClusters = []

        for i in range(0, len(center_n)):
            empty = np.zeros((img.shape[0], img.shape[1]))
            empty = np.uint8(empty) 
            custerContainer.append(empty)
            
        res2Grey = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        
        for y in range(0,res2Grey.shape[0]):
            for x in range(0,res2Grey.shape[1]):
                i = center_n.index(res2Grey[y][x])
                custerContainer[i][y][x] = 255

        for img in custerContainer:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.opening,self.opening))
            morph_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.closing,self.closing))
            morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel)
            
            finalClusters.append(morph_img)
            
        return finalClusters, res2Grey, custerContainer



    def findSuspects(self, clusters):
        p_defects = []

        for img in clusters:
            cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                if cv2.contourArea(c) >= self.c_size:
                    p_defects.append(c)
    
        return p_defects
    

    def inspect(self):
        
        clusters, res2Grey, custerContainer = self.cluster()
        suspects = self.findSuspects(clusters)

        tileWidth = 224
        tileHeight = 224
        tiles = []
        
        # print("running tiling "+str(len(suspects)))

        starts = []

        for c in suspects:
            # print("cluster started " + str(self.c_size))
            x,y,w,h = cv2.boundingRect(c)

            xStart = (int(x/tileWidth) * tileWidth)
            yStart = (int(y/tileHeight) * tileHeight)

            curY = yStart

            while curY < (y+h-1):
                curX = xStart
                while curX < (x+w-1):

                    # dont want tiles to exeed the image 
                    if (curX + tileWidth > self.img.shape[1]-1):
                        curX = self.img.shape[1]- 1 - tileWidth


                    if (curY + tileHeight > self.img.shape[0] - 1):
                        curY = self.img.shape[0] - 1 - tileHeight

                    
                    # tile = Tile(curX, curY, tileWidth, tileHeight, c)
                    # tile.roi = self.img[curY:curY+tileHeight, curX:curX+tileWidth]
                    # tiles.append(tile)

                    if [curX, curY] in starts:
                        i = starts.index([curX, curY])
                        tiles[i].suspects.append(c)
                    else:
                        #######################################
                        starts.append([curX, curY])
                        tile = Tile(curX, curY, tileWidth, tileHeight, c)
                        tile.roi = self.img[curY:curY+tileHeight, curX:curX+tileWidth]
                        tiles.append(tile)

                    curX = curX + tileWidth

                curY = curY + tileHeight

        return tiles, self.img, clusters

class CnnInspector(Inspector): 

    model = None

    def __init__(self, K, c_size, model, opening, closing, blur):
        Inspector.__init__(self, K, c_size, opening, closing, blur)
        self.model = model

    def setParams(self, K, c_size, model, opening, closing, blur):
        Inspector.setParams(self, K, c_size, opening, closing, blur)
        self.model = model


    def infer(self, tiles):
        indices = []
        predictions = []
        probailities = []
        
        for i in range(0, len(tiles)):
            pred,pred_idx,probs = self.model.predict(tiles[i].roi)
            if pred == "defect":
                indices.append(i)
                predictions.append(pred)
                probailities.append(probs[pred_idx])
                   
        return indices, predictions, probailities
    

    def inspect(self):
        # defectCnts = []
        defectTiles = []

        tiles, img, clusters = Inspector.inspect(self)
                
        indices, predictions, probailities = self.infer(tiles)
        p_count = 0
        for i in indices:
            # defectCnts.append(suspects[i])
            tiles[i].setDefect(probailities[p_count])
            p_count += 1
            defectTiles.append(tiles[i])
                     
        # return suspects, defectCnts, defectTiles, tiles, predictions, probailities, clusters, img
        return defectTiles, tiles, predictions, probailities, clusters, img

def addContours(img, tiles):
    contours = []
    for t in tiles:
        for c in t.suspects:
            contours.append(c)
    cv2.drawContours(img, contours, -1, (0,0,255), 3)

    
def loadModel():
    # return load_learner('../export.pkl', cpu=True)
    return load_learner('export.pkl', cpu=True)
