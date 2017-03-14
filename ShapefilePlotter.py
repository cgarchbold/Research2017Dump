import shapefile as shp
import matplotlib.pyplot as plt

sf = shp.Reader(r'C:/Users/Cohen/Desktop/Parcel/Parcel.shp')

plt.figure()
plt.show()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)
plt.show()
