from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class HeightfieldSpec:
    size: Tuple[float, float]
    field_res: Tuple[int, int]
    height_scale: float
    seed: int


def make_heightfield_xml(xml_text: str, spec: HeightfieldSpec) -> str:
    rng = np.random.default_rng(spec.seed)
    width, height = spec.field_res
    heights = rng.standard_normal(width * height).reshape(height, width)
    heights = np.clip(heights, -2.0, 2.0)
    heights = heights * spec.height_scale
    height_text = " ".join(f"{value:.5f}" for value in heights.flatten())

    root = ET.fromstring(xml_text)

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    hfield = ET.SubElement(asset, "hfield")
    hfield.set("name", "rough_field")
    hfield.set("size", f"{spec.size[0]} {spec.size[1]} {spec.height_scale} 0.1")
    hfield.set("nrow", str(height))
    hfield.set("ncol", str(width))
    hfield.text = height_text

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF missing worldbody")

    floor_geom = None
    for geom in worldbody.findall("geom"):
        if geom.get("name") == "floor":
            floor_geom = geom
            break
    if floor_geom is not None:
        worldbody.remove(floor_geom)

    rough = ET.SubElement(worldbody, "geom")
    rough.set("name", "rough_floor")
    rough.set("type", "hfield")
    rough.set("hfield", "rough_field")
    rough.set("pos", "0 0 0")
    rough.set("material", "grid")
    rough.set("contype", "1")
    rough.set("conaffinity", "1")

    return ET.tostring(root, encoding="unicode")
