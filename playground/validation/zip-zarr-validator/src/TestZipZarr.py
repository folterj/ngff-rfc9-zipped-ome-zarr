import json
import ome_zarr_models
from ome_zarr_models.base import BaseAttrs
import os.path
import re
import zarr
from zarr.storage import ZipStore
import zipfile

from util import check_for_zip64_signature


class TestZipZarr:
    metadata_filename = 'zarr.json'
    uri = 'C:/Project/slides/ozx/6001240.ozx'

    def __init__(self):
        assert zipfile.is_zipfile(self.uri), f'file not recognised as zip file'
        self.zip = zipfile.ZipFile(self.uri)
        self.zip_filenames = self.zip.namelist()
        self.store = ZipStore(self.uri)
        self.root = zarr.open(self.store, mode='r')
        self.metadata = self.root.metadata.to_dict()
        self.data = [self.root.get(node) for level, node in enumerate(self.root)]

    def zip_list_root(self):
        root_paths = set()
        for filename in self.zip_filenames:
            if '/' in filename:
                filename = filename[:filename.index('/')]
            root_paths.add(filename)
        return root_paths

    def test_requirement12(self):
        # The ZIP file MUST contain exactly one OME-Zarr hierarchy.
        # The root of the ZIP archive MUST correspond to the root of the OME-Zarr hierarchy. The ZIP file MUST contain the OME-Zarr’s root-level zarr.json.
        assert isinstance(self.metadata, dict), f'metadata is not a dict: {self.metadata}'
        model = ome_zarr_models.open_ome_zarr(self.root) # this function validates the zarr
        assert isinstance(model.ome_attributes, BaseAttrs), f'Invalid zarr'
        root_zarr_filenames = set([self.metadata_filename] + list(self.root.keys()))
        root_zip_filenames = self.zip_list_root()
        assert root_zarr_filenames == root_zip_filenames, f'Root hierarchy invalid'

    def test_requirement3(self):
        # OME-Zarr zip files SHALL NOT be embedded in a parent OME-Zarr hierarchy (as a sub-hierarchy or otherwise).
        assert '.zar' not in os.path.dirname(self.uri).lower()

    def test_requirement4(self):
        # OME-Zarr zip files SHALL NOT be split into multiple parts.
        # (ZipStore uses zipfile, which does not currently support multi-part zip files (2025))
        assert not os.path.splitext(self.uri)[1].lstrip('.').isdigit(), f'Multi-part files not allowed'

    def test_recommendation1(self):
        # The ZIP64 format extension SHOULD be used, irrespective of the ZIP file size.
        assert check_for_zip64_signature(self.uri) == True, 'ZIP64 format extension should be used'

    def test_recommendation2(self):
        # ZIP-level compression SHOULD be disabled in favor of Zarr-level compression codecs.
        zip = self.zip
        assert zip.compression == zipfile.ZIP_STORED, f'Compression should be disabled, using "STORE" instead of {zip.compression}'
        assert zip.compresslevel is None, f'Compresslevel should be None instead of {zip.compresslevel}'

    def test_recommendation3(self):
        # The sharding codec SHOULD be used to reduce the number of entries within the ZIP archive.
        assert self.data[0].metadata.shards, f'No sharding'

    def test_recommendation4(self):
        # The root-level zarr.json file SHOULD be the first ZIP file entry and the first entry in the central directory header; other zarr.json files SHOULD follow immediately afterwards, in breadth-first order.
        assert self.zip_filenames[0] == self.metadata_filename, f'{self.metadata_filename} found after other file'
        seen_non_zarrjson = False
        for filename in self.zip_filenames:
            is_zarrjson = (os.path.basename(filename) == self.metadata_filename)
            if not is_zarrjson:
                seen_non_zarrjson = True
            elif seen_non_zarrjson:
                assert False, f'{self.metadata_filename} found after other file'

    def test_recommendation5(self):
        # The ZIP archive comment SHOULD contain null-terminated UTF-8-encoded JSON with an ome attribute that holds a version key with the OME-Zarr version as string value, equivalent to {"ome": { "version": "XX.YY" }}.
        comment = json.loads(self.zip.comment.decode("utf-8").replace("'", '"'))
        assert isinstance(comment, dict), f'metadata dictionary comment expected instead of "{comment}"'
        assert 'ome' in comment, f'[ome] in comment dict expected'
        assert 'version' in comment['ome'], f'[ome][version] in comment dict expected'
        assert re.fullmatch(r'\d.\d', comment['ome']['version']) is not None, f'version: "XX.YY" in comment dict expected instead of {comment['ome']['version']}'

    def test_recommendation6(self):
        # The name of OME-Zarr zip files SHOULD end with .ozx.
        assert self.uri.endswith('.ozx'), f'{self.uri} should end with .ozx'

    def test_zarr_read(self):
        assert isinstance(self.metadata, dict)
        assert isinstance(self.data, list)
        assert self.data[0].shape is not None


if __name__ == "__main__":
    validator = TestZipZarr()

    validator.test_requirement12()
    validator.test_requirement3()
    validator.test_requirement4()

    validator.test_recommendation1()
    validator.test_recommendation2()
    validator.test_recommendation3()
    validator.test_recommendation4()
    validator.test_recommendation5()
    validator.test_recommendation6()

    validator.test_zarr_read()
