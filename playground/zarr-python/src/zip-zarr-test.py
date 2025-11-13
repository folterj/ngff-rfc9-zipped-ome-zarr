import numpy as np
from ome_zarr.scale import Scaler
import zarr
from zarr.storage import ZipStore


def test_zarr_read(uri):
    store = ZipStore(uri)
    root = zarr.open(store, mode='r')
    metadata = root.metadata.to_dict()
    data = [root.get(node) for level, node in enumerate(root)]
    return metadata, data


def test_zarr_write(uri, data, dim_order, pixel_size_um):
    pyramid_datas = []
    zarr_datas = []
    store = ZipStore(uri, mode='w')
    scaler = Scaler()

    datasets = []
    scale = 1
    for level in range(1 + scaler.max_layer):
        datasets.append({
            'path': str(level),
            'coordinateTransformations': create_transformation_metadata(dim_order, pixel_size_um, scale)
        })
        scale /= scaler.downscale

    multiscales = {
        'version': 0.5,
        'axes': create_axes_metadata(dim_order),
        'name': uri,
        'datasets': datasets,
    }
    metadata = {'multiscales': [multiscales]}

    # Avoid re-creating root metdata, by providing all metadata in creation of root group
    root = zarr.create_group(store, attributes=metadata)

    for level in range(1 + scaler.max_layer):
        if level > 0:
            data = scaler.resize_image(data)
        # Using write_data=False only writes metadata; store zarr array & pyramid data for later
        zarr_data = root.create_array(name=str(level), data=data, chunks=(10, 10), write_data=False)
        zarr_datas.append(zarr_data)
        pyramid_datas.append(data)

    # Now write pyramid data into zarr arrays
    for zarr_data, pyramid_data in zip(zarr_datas, pyramid_datas):
        zarr_data[:] = pyramid_data


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
        axis = {'name': dim, 'type': type1}
        if unit1 is not None and unit1 != '':
            axis['unit'] = unit1
        axes.append(axis)
    return axes


def create_transformation_metadata(dim_order, pixel_size_um, scale, translation_um={}):
    metadata = []
    pixel_size_scale = []
    translation_scale = []
    for dim in dim_order:
        if dim in pixel_size_um:
            pixel_size_scale1 = pixel_size_um[dim]
        else:
            pixel_size_scale1 = 1
        if dim in 'xy':
            pixel_size_scale1 /= scale
        pixel_size_scale.append(pixel_size_scale1)

        if dim in translation_um:
            translation1 = translation_um[dim]
        else:
            translation1 = 0
        if dim in 'xy':
            pixel_size_scale1 *= scale
        translation_scale.append(translation1)

    metadata.append({'type': 'scale', 'scale': pixel_size_scale})
    if not all(v == 0 for v in translation_scale):
        metadata.append({'type': 'translation', 'translation': translation_scale})
    return metadata


if __name__ == "__main__":
    #filename = 'C:/Project/slides/6001240.zarr'
    #filename = 'C:/Project/slides/ozx/6001240.ozx'
    #result = test_zarr_read(filename)
    #print(result)

    filename = 'C:/Project/slides/ozx/test.ozx'
    data = np.random.rand(100, 100)
    dim_order = 'yx'
    pixel_size = {'x': 1, 'y': 1}
    test_zarr_write(filename, data, dim_order, pixel_size)

    result = test_zarr_read(filename)
    print(result)
