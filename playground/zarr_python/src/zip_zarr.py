import json
import numpy as np
from ome_zarr.scale import Scaler
import zarr
from ome_zarr_models.v05 import Image
from ome_zarr_models.v05.axes import Axis
from pydantic_zarr.v3 import ArraySpec
from zarr.storage import ZipStore
from zipfile import ZipFile


def zip_zarr_read(uri):
    store = ZipStore(uri)
    root = zarr.open(store, mode='r')
    metadata = root.metadata.to_dict()['attributes']['ome']
    data = get_zarr_data(root)
    return metadata, data


def get_zarr_data(group):
    data = []
    for level, node in enumerate(group):
        node = group.get(node)
        if isinstance(node, zarr.Group):
            data.extend(get_zarr_data(node))
        else:
            data.append(node)
    return data


def zip_zarr_write(uri, data, dim_order, pixel_size_um):
    pyramid_datas = []
    zarr_datas = []
    store = ZipStore(uri, mode='w')
    scaler = Scaler()
    downscale = scaler.downscale

    scales, transforms = [], []
    paths = []
    scale = 1
    for level in range(1 + scaler.max_layer):
        if level > 0:
            data = scaler.resize_image(data)
        pyramid_datas.append(data)
        paths.append(str(level))
        scales1, transforms1 = create_transformation_metadata(dim_order, pixel_size_um, scale)
        scales.append(scales1)
        transforms.append(transforms1)
        scale /= downscale

    array_specs = [ArraySpec.from_array(data, dimension_names=list(dim_order)) for data in pyramid_datas]

    ome_zarr_image = Image.new(
        array_specs=array_specs,
        paths=paths,
        axes=create_axes_metadata(dim_order),
        scales=scales,
        translations=transforms,
    )

    ome_zarr_attributes = ome_zarr_image.model_dump()['attributes']

    # Avoid re-creating root metdata, by providing all metadata in creation of root group
    root = zarr.create_group(store, attributes=ome_zarr_attributes)

    for level in range(1 + scaler.max_layer):
        # Using write_data=False only writes metadata; store zarr array & pyramid data for later
        zarr_data = root.create_array(name=str(level), dimension_names=list(dim_order),
                                      data=data, chunks=(10, 10), shards=(10, 10), write_data=False)
        zarr_datas.append(zarr_data)

    # Now write pyramid data into zarr arrays
    for zarr_data, pyramid_data in zip(zarr_datas, pyramid_datas):
        zarr_data[:] = pyramid_data

    store.close()

    with ZipFile(uri, 'a') as zipfile1:
        zipfile1.comment = json.dumps({'ome': {'version': ome_zarr_attributes['ome']['version']}}).encode('utf-8')


def create_axes_metadata(dim_order):
    axes = []
    for dim in dim_order:
        unit1 = None
        if dim == 't':
            type1 = 'time'
            unit1 = 'millisecond'
        elif dim == 'c':
            type1 = 'channel'
        else:
            type1 = 'space'
            unit1 = 'micrometer'
        if unit1 is not None and unit1 != '':
            axis = Axis(name=dim, type=type1, unit=unit1)
        else:
            axis = Axis(name=dim, type=type1)
        axes.append(axis)
    return axes


def create_transformation_metadata(dim_order, pixel_size_um, scale, translation_um={}):
    scales = []
    translations = []
    for dim in dim_order:
        if dim in pixel_size_um:
            pixel_size_scale1 = pixel_size_um[dim]
        else:
            pixel_size_scale1 = 1
        if dim in 'xy':
            pixel_size_scale1 /= scale
        scales.append(pixel_size_scale1)

        if dim in translation_um:
            translation1 = translation_um[dim]
        else:
            translation1 = 0
        if dim in 'xy':
            pixel_size_scale1 *= scale
        translations.append(translation1)

    return scales, translations


if __name__ == "__main__":
    #filename = 'C:/Project/slides/6001240.zarr'
    #filename = 'C:/Project/slides/ozx/6001240.ozx'
    #filename = 'C:/Project/slides/ozx/kingsnake.ozx'
    #result = zip_zarr_read(filename)
    #print(result)

    filename = 'C:/Project/slides/ozx/test.ozx'
    data = np.random.rand(100, 100)
    dim_order = 'yx'
    pixel_size = {'x': 1, 'y': 1}
    zip_zarr_write(filename, data, dim_order, pixel_size)

    result = zip_zarr_read(filename)
    print(result)
