"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Data API client
"""
import base64
import io
import json
import requests

import affine

from scipy import sparse
from numpy import ndarray, frombuffer

from climada.entity import BlackMarble
from climada.entity import Tag as EntTag

from climada.hazard import TropCyclone
from climada.hazard import Tag as HazTag
from climada.hazard import Centroids


def deserialize_ndarray(data: dict) -> ndarray:

    return frombuffer(base64.b64decode(data['bytes']), data['dtype'])


def deserialize_csr_matrix(data: str) -> sparse.csr_matrix:
    byt = base64.b64decode(data)
    with io.BytesIO(byt) as bio:
        return sparse.load_npz(bio)


def tc_from_jsonble(jo: dict) -> TropCyclone:
    jo = jo['hazard']
    assert jo['type'] == 'TropCyclone'

    tc = TropCyclone()

    for ak, av in jo['attributes'].items():
        if ak == 'tag':
            tc.tag = tag_from_jsonble(av)
        else:
            setattr(tc, ak, av)

    tc.centroids = centroids_from_jsonble(jo['data']['centroids'])
    for lst in ['event_name', 'basin']:
        setattr(tc, lst, jo['data'][lst].split(','))
    for nda in ['event_id', 'frequency', 'date', 'orig', 'category']:
        setattr(tc, nda, deserialize_ndarray(jo['data'][nda]))
    for csr in ['intensity', 'fraction']:
        setattr(tc, csr, deserialize_csr_matrix(jo['data'][csr]))
    return tc


def meta_from_jsonble(jo: dict) -> dict:
    meta = dict()
    meta.update(jo)
    meta['transform'] = affine.Affine(**jo['transform'])
    return meta


def tag_from_jsonble(jo: dict) -> HazTag:
    return HazTag(**jo)


def centroids_from_jsonble(jo: dict) -> Centroids:
    c = Centroids()
    setattr(c, 'meta', meta_from_jsonble(jo['meta']))
    for attr in ['lat', 'lon', 'area_pixel', 'dist_coast', 'on_land', 'region_id', 'elevation']:
        setattr(c, attr, deserialize_ndarray(jo[attr]))
    # setattr(c, 'geometry', ...)
    return c


def black_marble_from_jsonble(jo: dict) -> BlackMarble:

    jo = jo['exposure']
    assert jo['type'] == 'BlackMarble'

    bm = BlackMarble(dict([
        (k, deserialize_ndarray(v))
        for (k, v) in jo['data'].items()
    ]))
    for ak, av in jo['attributes'].items():
        setattr(bm, ak, av)
    bm.tag = EntTag(**bm.tag)
    bm.meta['transform'] = affine.Affine(**bm.meta['transform'])
    return bm


class DataApiClient(object):
    def __init__(self, host='https://climada.ethz.ch'):
        self.host = host

    def get_trop_cyclone(self, **kwargs):

        url = f'{self.host}/rest/hazard/tropcyclone?' \
            + "&".join([f'{k}={v}' for (k, v) in kwargs.items()])

        w = requests.get(url)
        jo = json.loads(w.content.decode())

        return tc_from_jsonble(jo)

    def get_black_marble(self, **kwargs):

        url = f'{self.host}/rest/exposure/blackmarble?' \
            + "&".join([f'{k}={v}' for (k, v) in kwargs.items()])

        w = requests.get(url)
        jo = json.loads(w.content.decode())

        return black_marble_from_jsonble(jo)
