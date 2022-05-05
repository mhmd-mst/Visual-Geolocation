
"""
You can use this script to download images from Flickr within a given area.
This script works in 3 steps:
    1) First find all IDs of flickr images within the given area.
    2) For each ID find its coordinates and the URLs to download the image.
    3) Finally download the images, saving the coordinates and the flickr_id in the name.

Example to run this script:
$ python download_flickr.py --min_lat 37.80 --max_lat 37.804 --min_lon -122.414 --max_lon -122.41

"""

import os
import re
import sys
import utm
import torch
import random
import shutil
import argparse
import traceback
import numpy as np
from tqdm import tqdm
import urllib.request
import multiprocessing
from datetime import datetime
from urllib.request import urlopen


def get_whole_lines(pattern, lines):
    return [l for l in lines if re.search(pattern, l)]


def get_matches(pattern, lines):
    lines = get_whole_lines(pattern, lines)
    return [re.search(pattern, l).groups()[0] for l in lines]


def parallelize_function(processes_num, target_function, list_object, *args):
    """For each process take a sublist out of list_object and pass it to target_function."""
    assert type(list_object) == list, f"in parallelize_function() list_object must be a list, but it's a {type(list_object)}"
    jobs = []
    processes_num = min(processes_num, len(list_object))
    sublists = np.array_split(list_object, processes_num)
    # The first process uses tqdm
    sublists[0] = tqdm(sublists[0], ncols=80)
    for process_num in range(processes_num):
        all_arguments = (process_num, processes_num, sublists[process_num], *args)
        p = multiprocessing.Process(target=target_function,
                                    args=all_arguments)
        jobs.append(p)
        p.start()
    for proc in jobs: proc.join()


def format_coord(num, left=2, right=5):
    """Return the formatted number as a string with (left) int digits 
            (including sign '-' for negatives) and (right) float digits.
    >>> format_coord(1.1, 3, 3)
    '001.100'
    >>> format_coord(-0.123, 3, 3)
    '-00.123'
    """
    sign = "-" if float(num) < 0 else ""
    num = str(abs(float(num))) + "."
    integer, decimal = num.split(".")[:2]
    left -= len(sign)
    return f"{sign}{int(integer):0{left}d}.{decimal[:right]:<0{right}}"


def format_location_info(latitude, longitude):
    easting, northing, zone_number, zone_letter = utm.from_latlon(float(latitude), float(longitude))
    easting = format_coord(easting, 7, 2)
    northing = format_coord(northing, 7, 2)
    latitude = format_coord(latitude, 3, 5)
    longitude = format_coord(longitude, 4, 5)
    return easting, northing, zone_number, zone_letter, latitude, longitude


def is_valid_timestamp(timestamp):
    """Return True if it's a valid timestamp, in format YYYYMMDD_hhmmss,
        with all fields from left to right optional.
    >>> is_valid_timestamp('')
    True
    >>> is_valid_timestamp('201901')
    True
    >>> is_valid_timestamp('20190101_123000')
    True
    """
    return bool(re.match("^(\d{4}(\d{2}(\d{2}(_(\d{2})(\d{2})?(\d{2})?)?)?)?)?$", timestamp))


def get_dst_image_name(latitude, longitude, pano_id=None, tile_num=None, heading=None,
                       pitch=None, roll=None, height=None, timestamp=None, note=None, extension=".jpg"):
    easting, northing, zone_number, zone_letter, latitude, longitude = format_location_info(latitude, longitude)
    tile_num  = f"{int(float(tile_num)):02d}" if tile_num  is not None else ""
    heading   = f"{int(float(heading)):03d}"  if heading   is not None else ""
    pitch     = f"{int(float(pitch)):03d}"    if pitch     is not None else ""
    roll      = f"{int(float(roll)):03d}"     if roll      is not None else ""
    height    = f"{int(float(height)):03d}"   if height    is not None else ""
    timestamp = f"{timestamp}"                if timestamp is not None else ""
    note      = f"{note}"                     if note      is not None else ""
    assert is_valid_timestamp(timestamp), f"{timestamp} is not in YYYYMMDD_hhmmss format"
    
    return f"@{easting}@{northing}@{zone_number:02d}@{zone_letter}@{latitude}@{longitude}" + \
           f"@{pano_id}@{tile_num}@{heading}@{pitch}@{roll}@{height}@{timestamp}@{note}@{extension}"


###############################################################################
###############################################################################
###############################################################################
# /bin/python3 /home/hajali/Desktop/DS/MLDL/Visual-Geolocation/download_flickr.py --min_lat 37.7 --max_lat 37.82 --min_lon -122.53 --max_lon -122.35
# python download_flickr.py --min_lat 37.7 --max_lat 37.82 --min_lon -122.53 --max_lon -122.35

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--min_lat", type=float, required=False, default= 37.74)
parser.add_argument("--max_lat", type=float, required=False, default=37.78)
parser.add_argument("--min_lon", type=float, required=False, default=-122.51)
parser.add_argument("--max_lon", type=float, required=False, default=-122.45)
parser.add_argument("--processes_num", type=int, default=8, help="_")
parser.add_argument("--output_folder", type=str, default="outputs",
                    help="Folder where to save logs and maps")

args = parser.parse_args()
args.output_folder = (f"{args.output_folder}/" +
                      f"{args.min_lat}_{args.max_lat}_{args.min_lon}_{args.max_lon}")
flickr_folder = f"{args.output_folder}/flickr"
print(" ".join(sys.argv))
print(args)
images_folder = f"{args.output_folder}/flickr/images"
os.makedirs(images_folder, exist_ok=True)

manager = multiprocessing.Manager()


#### First find all IDs of flickr images within the given area.
print("First find all IDs of flickr images within the given area")

def search_flickr_ids_in_bbox(flickr_ids_, lat_, lon_, side_len_):
    url = ("https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key=181073bd1c1766c272d30a9623f96e57&format=rest" +
           f"&bbox={lon_-side_len_}%2C{lat_-side_len_}%2C{lon_}%2C{lat_}&per_page=100")
    for trial_num in range(10):
        try:
            lines = urlopen(url).read().decode('utf-8').split("\n")
            pages_num = int(get_matches("pages=\"(\d+)\"", lines)[0])
            for page_num in range(1, pages_num+2):
                paged_url = url + f"&page={page_num}"
                lines = urlopen(paged_url).read().decode('utf-8').split("\n")
                for flickr_id in get_matches("id=\"(\d+)\" owner=", lines):
                    flickr_ids_[flickr_id] = None
            print(pages_num)
            break
        except urllib.error.HTTPError as e:
            print(f"lat: {lat_}, lon: {lon_}; Exception: {e}")
        except Exception as e:
            print(f"lat: {lat_}, lon: {lon_}; Exception: {e}")

def download_flickr_ids(process_num, processes_num, all_lats_lons_sublist, flickr_ids, side_len):
    for lat, lon in all_lats_lons_sublist:
        search_flickr_ids_in_bbox(flickr_ids, lat, lon, side_len)

try:
    torch.load(f"{flickr_folder}/flickr_ids.torch")
except:

    # It is necessary to query small areas (i.e. with short side_len), because flickr's APIs have bugs.
    # If we query the whole area instead of smaller sub-areas, we'd get only at most few 1000 images.
    # side_len = 0.001
    side_len = 0.015
    # decimals_num = 3
    decimals_num = 2
    # flickr_ids is a dict but it is used as a set. This is because there is no manager.set() class
    flickr_ids = manager.dict()

    lats = np.arange(args.min_lat, args.max_lat, side_len)
    lons = np.arange(args.min_lon, args.max_lon, side_len)
    all_lats_lons = []
    for lat in lats:
        for lon in lons:
            all_lats_lons.append((round(lat, decimals_num), round(lon, decimals_num)))

    random.shuffle(all_lats_lons)  # Shuffle because some areas are dense with images.

    parallelize_function(args.processes_num, download_flickr_ids, all_lats_lons, flickr_ids, side_len)

    flickr_ids = sorted(list(flickr_ids.keys()))
    torch.save(flickr_ids, f"{flickr_folder}/flickr_ids.torch")
else:
    flickr_ids = torch.load(f"{flickr_folder}/flickr_ids.torch")

print(f"I found {len(flickr_ids)} IDs (aka photos) in this area")


#### For each ID find its coordinates and the URLs to download the image.
print("For each ID find its coordinates and the URLs to download the image")

def download_images_metadata(process_num, processes_num, flickr_ids_sublist, dict__flickr_id__info):
    for i, flickr_id in enumerate(flickr_ids_sublist):
        try:
            print("Flickr ID: ", flickr_id)
            response = urllib.request.urlopen(f"http://flickr.com/photo.gne?id={flickr_id}").read().decode('utf-8')
            # Each image can be downloaded from many URLs (image_urls). Each URL results in the same image with different resolution.
            image_urls = list(set(re.findall(fr'(live.staticflickr.com\/.*?\/{flickr_id}_.*?.jpg)', response)))
            image_urls = [u for u in image_urls if re.compile(".+_..jpg").match(u)]
            # Sort the available URLs by placing first the ones with highest resolution (only the first URL will be used).
            image_urls = sorted(image_urls, key=lambda x: ["o", "b", "c", "z", "m", "w", "n", "t"].index(x.split(".jpg")[0][-1]))
            print("URLs: ", image_urls)
            
            lat, lon = re.findall(r'"latitude":(.*?),"longitude":(.*?),"accuracy"', response)[0]
            date = re.findall(r'<span class="date-taken-label" title="Uploaded on (.*)">', response)[0]
            timestamp = datetime.strptime(date, '%B %d, %Y').strftime("%Y%m%d")  # timestamp in format YYYYMMDD
            dict__flickr_id__info[flickr_id] = (lat, lon, image_urls, timestamp)
            # print("Dict FLickr ID: ", dict__flickr_id__info)

        except Exception as e:
            print(f"flickr_id: {flickr_id}, i: {i}, Exception: {e}")
            print("traceback: " + traceback.format_exc())

dict__flickr_id__info = manager.dict()

parallelize_function(args.processes_num, download_images_metadata, flickr_ids, dict__flickr_id__info)

dict__flickr_id__info = dict__flickr_id__info.copy()
torch.save(dict__flickr_id__info, f"{flickr_folder}/dict__flickr_id__info.torch")


#### Finally download the images, saving the coordinates and the flickr_id in the name.
print("Finally download the images, saving their coordinates and ID in the name")

def download_images(process_num, processes_num, sublist__flickr_id__info):
    for i, (flickr_id, (lat, lon, image_urls, timestamp)) in enumerate(sublist__flickr_id__info):
        try:
            dst_image_name = get_dst_image_name(lat, lon, pano_id=flickr_id, timestamp=timestamp)
            _ = urllib.request.urlretrieve(f"https://{image_urls[0]}", f".{dst_image_name}")
            shutil.move(f".{dst_image_name}", f"{images_folder}/{dst_image_name}")
        except Exception as e:
            print(f"flickr_id: {flickr_id}, i: {i}, Exception: {e}")
            print("traceback: " + traceback.format_exc())

list__flickr_id__info = sorted(list(dict__flickr_id__info.items()))

parallelize_function(args.processes_num, download_images, list__flickr_id__info)

# --min_lat 37.76 --max_lat 37.8 --min_lon -122.46 --max_lon -122.40
