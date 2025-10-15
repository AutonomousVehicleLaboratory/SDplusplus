'''
usage: from codes run 
$ python get_HD_map_av2.py --map_name d9910975-9eaa-48df-92b7-c72ac01c17a3 --llm_model gpt-4o-mini
'''
import json
import math
import time
import argparse
import xml.etree.ElementTree as ET
import math
from math import radians, cos, sin, asin, sqrt

parser = argparse.ArgumentParser()
parser.add_argument('--map_name', type=str)
parser.add_argument('--scale', type=float, default=20)
# parser.add_argument('--llm_type', type=str, default='openai')
parser.add_argument('--llm_type', type=str, default='llama')
# parser.add_argument('--llm_type', type=str, default='deepseek')
parser.add_argument('--llm_model', type=str, default=None)
args = parser.parse_args()


def get_osm_bounds(osm_file):
    """
    Extracts the bounding box (min/max latitude and longitude) from an OSM file.
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # First, check if <bounds> exists in the OSM file
    bounds = root.find("bounds")
    if bounds is not None:
        min_lat = float(bounds.attrib["minlat"])
        max_lat = float(bounds.attrib["maxlat"])
        min_lon = float(bounds.attrib["minlon"])
        max_lon = float(bounds.attrib["maxlon"])
        return min_lat, max_lat, min_lon, max_lon

    # If <bounds> tag doesn't exist, compute manually from nodes
    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")

    for node in root.findall("node"):
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])

        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)

    return min_lat, max_lat, min_lon, max_lon


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def convert_ll_to_ab_np(lat, lon, minlat, maxlat, minlon, maxlon):
    h = haversine(minlon, minlat, minlon, maxlat) * 1e7
    w = haversine(minlon, minlat, maxlon, minlat) * 1e7
    X = ((lon - minlon) / (maxlon - minlon)) * w
    Y = h - ((lat - minlat) / (maxlat - minlat)) * h
    return (X, Y)


def generate_hd_map(centerline_json, basic_road_info, road_info):
    def offset_point(point, distance, angle):
        x, y = point
        dx = distance * math.cos(angle)
        dy = distance * math.sin(angle)
        return [x + dx, y + dy]

    def generate_parallel_line(centerline, offset):
        parallel_line = []
        for i in range(len(centerline) - 1):
            x1, y1 = centerline[i]
            x2, y2 = centerline[i + 1]
            angle = math.atan2(y2 - y1, x2 - x1)
            perpendicular_angle = angle + math.pi / 2
            parallel_line.append(offset_point((x1, y1), offset, perpendicular_angle))
        parallel_line.append(offset_point((x2, y2), offset, perpendicular_angle))
        return parallel_line

    def feet_to_ABS(feet):
        osm_path = f'../osm_maps/{args.map_name}/scenario_{args.map_name}_r_200.osm'
        min_lat, max_lat, min_lon, max_lon = get_osm_bounds(osm_path)

        r_earth = 6378137
        pi = math.pi
        new_lat = min_lat + ((feet * 0.3048) / r_earth) * (180 / pi)

        X1, Y1 = convert_ll_to_ab_np(min_lat, min_lon, min_lat, max_lat, min_lon, max_lon)
        X2, Y2 = convert_ll_to_ab_np(new_lat, min_lon, min_lat, max_lat, min_lon, max_lon)

        return math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)


    hd_map = {}

    widths_view = set()

    # TODO: make changes to incorporate different number of bike lanes
    # breakpoint()
    for road in centerline_json["streets"]["roads"]:
        centerline = road[1]["center_line"]
        centerline = [[pt['x'], pt['y']] for pt in centerline['pts']]
        # road_id = str(road[1]["osm_ids"][0])
        road_id = str(road[1]["id"])
        if road_id not in list(road_info.keys()):
            print('Key not found')
            continue
        if road_info[road_id]["lane_width"] is None:
            lane_width = 0
        else:
            lane_width = feet_to_ABS(road_info[road_id]["lane_width"])
            # lane_width = 30000 # A/B Street Baseline
            widths_view.add((road_info[road_id]["lane_width"], lane_width))
        if road_info[road_id]["bike_lane_width"] is None:
            bike_lane_width = 0
        else:
            bike_lane_width = feet_to_ABS(feet_to_ABS(road_info[road_id]["bike_lane_width"]))
            # bike_lane_width = 15000 # A/BStreet Baseline
        total_width = lane_width * basic_road_info[road_id]["num_lanes"]/2 + bike_lane_width

        lanes = []
        for i in range(1, basic_road_info[road_id]["num_lanes"]):
            offset = (i - basic_road_info[road_id]["num_lanes"] / 2) * lane_width
            lanes.append(generate_parallel_line(centerline, offset))

        hd_road = {
            "road_name": road[1]["name"],
            "road_type": road[1]["highway_type"],
            "centerline": centerline,
            "lanes": lanes,
        }

        left_bike_lane_offset = -total_width + bike_lane_width
        right_bike_lane_offset = total_width - bike_lane_width

        if bike_lane_width == 0:
            hd_road["left_road_boundary"] = generate_parallel_line(centerline, -total_width)
            hd_road["right_road_boundary"] = generate_parallel_line(centerline, total_width)
        elif len(basic_road_info[road_id]["bike_lane_dir"]) == 0:
            hd_road["left_road_boundary"] = generate_parallel_line(centerline, -total_width + bike_lane_width)
            hd_road["right_road_boundary"] = generate_parallel_line(centerline, total_width - bike_lane_width)
        elif len(basic_road_info[road_id]["bike_lane_dir"]) == 1:
            if basic_road_info[road_id]["bike_lane_dir"][0] == "Fwd":
                hd_road["right_bike_lane"] = generate_parallel_line(centerline, right_bike_lane_offset)
                hd_road["right_road_boundary"] = generate_parallel_line(centerline, total_width)
                hd_road["left_road_boundary"] = generate_parallel_line(centerline, -total_width + bike_lane_width)
            else:
                hd_road["left_bike_lane"] = generate_parallel_line(centerline, left_bike_lane_offset)
                hd_road["left_road_boundary"] = generate_parallel_line(centerline, -total_width)
                hd_road["right_road_boundary"] = generate_parallel_line(centerline, total_width - bike_lane_width)
        else:
            hd_road["right_bike_lane"] = generate_parallel_line(centerline, right_bike_lane_offset)   
            hd_road["left_bike_lane"] = generate_parallel_line(centerline, left_bike_lane_offset)
            hd_road["left_road_boundary"] = generate_parallel_line(centerline, -total_width)
            hd_road["right_road_boundary"] = generate_parallel_line(centerline, total_width)

        hd_map[road_id] = hd_road

    return hd_map

output_dir = '../outputs/av2'


with open("{}/{}_output.json".format(output_dir, args.map_name), 'r') as f:
    centerline_json = json.load(f)

basic_road_info_path = '{}/{}_basic_road_info.json'.format(output_dir, args.map_name)
with open(basic_road_info_path,'r') as f:
    basic_road_info = json.load(f)

basic_road_info_osm = {}
for road_id in basic_road_info:
    osm_id = basic_road_info[road_id]['osm_ids'][0]
    basic_road_info_osm[str(osm_id)] = basic_road_info[road_id]

road_info_path = '{}/{}_road_info_{}_{}.json'.format(output_dir, args.map_name, args.llm_type, args.llm_model)
with open(road_info_path,'r') as f:
    road_info = json.load(f)

start = time.time()    
hd_map = generate_hd_map(centerline_json, basic_road_info, road_info)
print("Time taken for generating the HD Map: {:.2f} seconds".format(time.time() - start))
print(hd_map)

with open('{}/{}_{}_{}_hd_output.json'.format(output_dir, args.map_name, args.llm_type, args.llm_model), 'w') as f:
    json.dump(hd_map, f, indent=2)