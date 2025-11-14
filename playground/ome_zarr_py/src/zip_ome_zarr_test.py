import numpy as np
from ome_zarr import writer
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

import zarr
from zarr.storage import ZipStore


def _get_ome_zarr_reader(uri):
    location = parse_url(uri)
    if location is None:
        raise FileNotFoundError(f'Error parsing ome-zarr file {uri}')
    reader = Reader(location)
    nodes = list(reader())
    return reader, nodes


def zip_ome_zarr_read(filename, level=0):
    reader, nodes = _get_ome_zarr_reader(filename)
    # nodes may include images, labels etc
    if len(nodes) == 0:
        raise FileNotFoundError(f'No image data found in ome-zarr file {filename}')
    # first node will be the image pixel data
    image_node = nodes[0]
    metadata = image_node.metadata
    data = image_node.data[level]
    return metadata, data


def zip_ome_zarr_write(uri, data):
    store = ZipStore(uri, mode='w')
    root = zarr.create_group(store) # creates basic root zarr.json
    writer.write_image(image=data, group=root) # updates root zarr.json various times, writes pyramid metadata & data interlaced


def ome_zarr_write_zarr(uri, data):
    root = zarr.create_group(uri)
    writer.write_image(image=data, group=root)


if __name__ == "__main__":
    #filename = 'C:/Project/slides/6001240.zarr'
    #filename = 'C:/Project/slides/ozx/6001240.ozx'
    #result = zip_ome_zarr_read(filename)
    #print(result)

    filename = 'C:/Project/slides/ozx/test.ozx'
    data = np.random.rand(100, 100)
    zip_ome_zarr_write(filename, data)

    result = zip_ome_zarr_read(filename) # Doesn't work, unable to use external store
    print(result)
