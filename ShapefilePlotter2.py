import shapefile as shp
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import progressbar
from streetview_demo import StreetViewDownloader
import streetview_demo

sf = shp.Reader(r'/u/amo-d0/guest/cohen/StreetView2/Street.shp')



gg = StreetViewDownloader('./')
count=0
for shape in sf.iterShapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    for la, lo in zip(x, y):
	print list(gg.grabber([[la, lo]]))
    	print la, lo


