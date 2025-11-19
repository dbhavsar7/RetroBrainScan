# src/region_detector.py

import os
import json
import numpy as np


# Load brain region lookup file
def load_brain_regions():
    regions_path = os.path.join(os.path.dirname(__file__), "brain_regions.json")
    with open(regions_path, "r") as f:
        data = json.load(f)
    return data["spatial_map"]


# Slice CAM into 3 × 3 spatial zones
def divide_cam_into_zones(cam):
    """
    cam: np.array of shape (H, W) in [0,1]
    returns dict:
        {
            "top-left": intensity,
            "top-center": intensity,
            ...
        }
    """

    h, w = cam.shape
    h3 = h // 3
    w3 = w // 3

    zones = {
        "top-left": cam[0:h3, 0:w3],
        "top-center": cam[0:h3, w3:2*w3],
        "top-right": cam[0:h3, 2*w3:w],

        "mid-left": cam[h3:2*h3, 0:w3],
        "mid-center": cam[h3:2*h3, w3:2*w3],
        "mid-right": cam[h3:2*h3, 2*w3:w],

        "bottom-left": cam[2*h3:h, 0:w3],
        "bottom-center": cam[2*h3:h, w3:2*w3],
        "bottom-right": cam[2*h3:h, 2*w3:w],
    }

    # compute mean intensity for each region
    zone_intensity = {k: float(v.mean()) for k, v in zones.items()}
    return zone_intensity


# Get top anatomical regions from CAM
def extract_notable_regions(cam, top_n=2):
    """
    cam: numpy array (HxW)
    returns: list of region names
    """

    # load spatial mapping
    spatial_map = load_brain_regions()

    # compute intensities per spatial zone
    zone_scores = divide_cam_into_zones(cam)

    # sort zones by intensity (descending)
    sorted_zones = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)

    # take top N zones
    top_zones = [zone for zone, score in sorted_zones[:top_n]]

    # map zones → anatomical regions
    regions = []
    for zone in top_zones:
        if zone in spatial_map:
            regions.extend(spatial_map[zone])

    # unique + preserve order
    seen = set()
    unique_regions = []
    for r in regions:
        if r not in seen:
            unique_regions.append(r)
            seen.add(r)

    return unique_regions[:top_n]

