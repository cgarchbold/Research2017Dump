''' streetview: Implements Google StreetView static api.
'''
import os, csv
import io
from br import br
import numpy as np
from PIL import Image
from lxml import etree
from multiprocess import Pool
from random import shuffle, randint, uniform


class StreetViewDownloader:

    def __init__(self, outdir):
        self.xml_dir = outdir + 'xmls'
        self.img_dir = outdir + 'images'

        if not os.path.exists(self.xml_dir):
            os.makedirs(self.xml_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def get_xml(self, lat, lon):
        serv = 'http://cbk%d.google.com/' % randint(0,3)
        info_url = serv + 'cbk?output=xml&ll=%s,%s'

        try:
            xml = br.open(info_url % (lat, lon)).read()
        except Exception as e:
            xml = None
        return xml    

    def get_metadata(self, tree):
        if len(tree):
            lat = tree[0].get('lat')
            lon = tree[0].get('lng')
            pano_id = tree[0].get('pano_id')
            return float(lat), float(lon), pano_id

    def get_thumbnail(self, fname, width, height, pano_id):
        thumb_url = 'http://cbk0.google.com/cbk?output=thumbnail&w=%d&h=%d&panoid=%s'
        return br.retrieve(thumb_url % (width, height, pano_id), fname)

    def get_tile(self, tile_url):
        try:
            raw_data = br.open(tile_url).read()
            memory_file = io.BytesIO(raw_data)
            tile = np.array(Image.open(memory_file))
        except Exception as e:
            print("Tile error: %s" % tile_url)
            print(e)
            tile = np.array([]) 
        return tile

    def get_panorama(self, fname, pano_id, zoom_level=3):
        server_url = 'http://cbk%d.google.com/' % randint(0,3)
        pano_url = server_url + 'cbk?output=tile&panoid=%s&zoom=%d&x=%d&y=%d'
        zoom_sizes = {3:(7,4), 4:(13,7), 5:(26,13)}
        max_x, max_y = zoom_sizes[zoom_level]

        jobs = []
        for y in xrange(max_y):
            for x in xrange(max_x):
                tile_url = pano_url % (pano_id, zoom_level, x, y)
                jobs.append(tile_url)

        p = Pool(len(jobs))
        tiles = p.map(self.get_tile, jobs)
        p.close()

        if all(x.size for x in tiles):
            tiles = np.array(tiles)
            strips = []
            for y in xrange(max_y):
                strips.append(np.hstack(tiles[y*max_x:(y+1)*max_x,:,:,:]))
            pano = np.vstack(strips)
            pano = pano[0:1664, 0:3328]
        else:
            pano = np.array([])
        return pano


    def grabber(self, locs):
        for q_lon, q_lat in locs:
            xml = self.get_xml(q_lat, q_lon)
            if xml:
                tree = etree.XML(xml)
                if len(tree):
                    [lat, lon, pano_id] = self.get_metadata(tree)
                    fname = "{}_{}".format(lon, lat)

                    yield (fname, (lon, lat))

                    print("pano found: {},{} ({}) -> {}".format(lon, lat, pano_id, fname))

                    xml_fname = os.path.join(self.xml_dir, fname + '.xml')
                    if not os.path.exists(xml_fname):
                        f = open(xml_fname, 'w')
                        f.write(xml)
                        f.close()
                    pano_fname = os.path.join(self.img_dir, fname + '.jpg')
                    if not os.path.exists(pano_fname):
                        pano = self.get_panorama(fname, pano_id)
                        if pano.size:
                            Image.fromarray(pano).save(pano_fname)
                else:
                    print("\tpano not available: %s,%f" % (q_lat, q_lon))


if __name__ == "__main__":
    gg = StreetViewDownloader('./')
    found_points = list(gg.grabber([
				[-74.005394, 40.715345]
		]))
    print(found_points)
