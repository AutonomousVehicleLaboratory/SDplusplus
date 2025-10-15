import pandas as pd
import requests
import argparse
import statistics
import os
import json
import numpy as np

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
from enum import Enum, unique
from typing import Dict, Final, Tuple, Union
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interp1d

from av2.map.map_api import ArgoverseStaticMap
import av2.geometry.utm as utm
from av2.utils.typing import NDArrayFloat, NDArrayInt
from utm_to_gsm import convert_gps_to_utm, CityName, convert_city_coords_to_utm, convert_utm_to_city_coords 

from collections import defaultdict
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
from pyproj import Proj
import xml.etree.ElementTree as ET


'''
1. For each AV2 Lane, Get corresponding OSM roads
-> For each road, store the OSM key as key and av2 lane IDs as values
2. For each SD++ Road, get corresponding OSM Road
-> For each SD++ Road, store th corresponding OSM Road

3. Write coordinate conversion from AV2 to OSM
4. Write coordinate conversion from OSM to SD++
-> Get AV2 and SD++ lane centerlines and boundaries to the same frame

5. Create a boolean dictionary with AV2 lane IDs as keys and T/F as values

6. For every road ID, get all av2 lanes for the corresponding OSM ID
SD++ Lanes <-> OSM ID <-> AV2 Lanes
7. Run hungarian matching (sklearn) between SD++ Lanes and AV2 Lanes. For every mapped AV2 Lane, change to T in boolean dictionary

8. Repeat this for every road in SD++.
9> Find the average chamfer distance for lane centerlines and lane boundaries.
'''

# folder = 'GPT_IG_Context/' #
folder = ''
# folder = 'Llama3.3_IG_context/'
print(folder)


def interpolate_to_boundary(p1, p2, min_x, max_x, min_y, max_y):
    """
    Finds the intersection of the line segment (p1, p2) with the crop boundary.
    Ensures the new point is on the boundary.
    """
    x1, y1 = p1
    x2, y2 = p2

    intersections = []

    # Check intersection with vertical boundaries (min_x and max_x)
    if x2 != x1:
        # Compute intersection at min_x
        t = (min_x - x1) / (x2 - x1)
        if 0 <= t <= 1:
            y_intersect = y1 + t * (y2 - y1)
            if min_y <= y_intersect <= max_y:
                intersections.append((min_x, y_intersect))

        # Compute intersection at max_x
        t = (max_x - x1) / (x2 - x1)
        if 0 <= t <= 1:
            y_intersect = y1 + t * (y2 - y1)
            if min_y <= y_intersect <= max_y:
                intersections.append((max_x, y_intersect))

    # Check intersection with horizontal boundaries (min_y and max_y)
    if y2 != y1:
        # Compute intersection at min_y
        t = (min_y - y1) / (y2 - y1)
        if 0 <= t <= 1:
            x_intersect = x1 + t * (x2 - x1)
            if min_x <= x_intersect <= max_x:
                intersections.append((x_intersect, min_y))

        # Compute intersection at max_y
        t = (max_y - y1) / (y2 - y1)
        if 0 <= t <= 1:
            x_intersect = x1 + t * (x2 - x1)
            if min_x <= x_intersect <= max_x:
                intersections.append((x_intersect, max_y))

    # Return the first valid intersection (should be within bounds)
    if intersections:
        return intersections[0]
    return None

def filter_lanes_sdpp(sd_lane_centerlines_city, min_x, max_x, min_y, max_y):
    filtered_lanes = []

    for lane in sd_lane_centerlines_city:
        filtered_lane = []
        i = 0

        while i < len(lane) - 1:
            p1 = lane[i]
            p2 = lane[i + 1]

            in_p1 = (min_x <= p1[0] <= max_x) and (min_y <= p1[1] <= max_y)
            in_p2 = (min_x <= p2[0] <= max_x) and (min_y <= p2[1] <= max_y)

            if in_p1 and in_p2:
                # Both points inside, keep p1
                filtered_lane.append(p1)
            elif not in_p1 and not in_p2:
                # Both points outside, discard p1
                pass
            else:
                # One point is inside, one is outside -> Interpolate to boundary
                intersection = interpolate_to_boundary(p1, p2, min_x, max_x, min_y, max_y)
                if intersection is not None:
                    if in_p1:
                        # Keep p1 and add interpolated boundary point
                        filtered_lane.append(p1)
                        filtered_lane.append(intersection)
                    else:
                        # Add interpolated boundary point before p2
                        filtered_lane.append(intersection)

            i += 1

        # Add last point if it was inside the boundary
        if (min_x <= lane[-1][0] <= max_x) and (min_y <= lane[-1][1] <= max_y):
            filtered_lane.append(lane[-1])

        if len(filtered_lane) > 0:
            filtered_lanes.append(np.array(filtered_lane))

    return filtered_lanes


def get_hdmap_bounds(hd_map):
    """
    Extracts the bounding box (min/max latitude and longitude) from an Argoverse 2 HD Map.
    
    :param hdmap_file: Path to the Argoverse 2 HD Map JSON file
    :return: (min_lat, max_lat, min_lon, max_lon)
    """

    min_lat, min_lon = float("inf"), float("inf")
    max_lat, max_lon = float("-inf"), float("-inf")

    # Iterate through lane segments to extract coordinates
    driveable_areas = hd_map['drivable_areas']
    for lane in driveable_areas.keys():
        for point in driveable_areas[lane]['area_boundary']:
            coor = np.array([float(point['x']), float(point['y'])]) .reshape(1, -1) # Assuming (x, y) corresponds to (lon, lat)
            min_lat, max_lat = min(min_lat, coor[0,0]), max(max_lat, coor[0,0])
            min_lon, max_lon = min(min_lon, coor[0,1]), max(max_lon, coor[0,1])

    return min_lat, max_lat, min_lon, max_lon


def is_not_intersection(lane_segments, lane_id):
    '''Returns:
    - bool: True if the lane is NOT an intersection, False if it is.
    '''
    return not lane_segments.get(lane_id, {}).get("is_intersection", False)


def filtered_way_ids(osm_file):
    # Parse the OSM XML file
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Extract all 'way' IDs
    way_ids = [way.get('id') for way in root.findall('way')]

    return way_ids


def chamfer_distance(set1, set2):
    """
    Compute the Chamfer distance between two sets of points after equalizing them.
    """
    # breakpoint()
    if set1.shape[0] == 0 or set2.shape[0] == 0:
        return float('inf')  # Return a large distance if one of the sets is empty
    
    d_set1 = np.linalg.norm(set1[-1] - set1[0])
    d_set2 = np.linalg.norm(set2[-1] - set2[0])

    # Determine the larger size
    # max_points = max(set1.shape[0], set2.shape[0])
    max_points = 100

    if d_set1 > d_set2:
        set1 = interpolate_lane(set1, max_points)
        set2 = interpolate_lane(set2, max(2, int(100 * (d_set2 / d_set1))))
    else:
        set2 = interpolate_lane(set2, max_points)
        set1 = interpolate_lane(set1, max(2, int(100 * (d_set1 / d_set2))))
    
    set1 = interpolate_lane(set1, max_points)
    set2 = interpolate_lane(set2, max_points)

    # Compute Chamfer distance
    dist1 = np.min(cdist(set1, set2), axis=1)
    dist2 = np.min(cdist(set2, set1), axis=1)

    if d_set1 > d_set2:
        return np.mean(dist2)
    else:
        return np.mean(dist1)
 

def compute_cost_matrix(av2_lanes, sdpp_lanes):
    """
    Compute the cost matrix based on Chamfer distance between AV2 and SD++ lanes.
    Args:
        av2_lanes: List of NumPy arrays, each representing a lane in AV2.
        sdpp_lanes: List of NumPy arrays, each representing a lane in SD++.
    Returns:
        cost_matrix: A 2D NumPy array where cost_matrix[i, j] is the Chamfer distance
                     between av2_lanes[i] and sdpp_lanes[j].
    """
    m = len(av2_lanes)
    n = len(sdpp_lanes)
    cost_matrix = np.zeros((m, n))
 
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = chamfer_distance(av2_lanes[i], sdpp_lanes[j])
    
    return cost_matrix


def find_optimal_matching(av2_lanes, sdpp_lanes):
    """
    Find the optimal matching between AV2 and SD++ lanes based on Chamfer distance.
    Args:
        av2_lanes: List of NumPy arrays, each of shape (N, 2).
        sdpp_lanes: List of NumPy arrays, each of shape (M, 2).
    Returns:
        matches: List of tuples (i, j), where i is the index of the AV2 lane,
                 and j is the index of the SD++ lane it is matched to.
    """
    # Compute the cost matrix
    cost_matrix = compute_cost_matrix(av2_lanes, sdpp_lanes)

    # Apply the Hungarian algorithm to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter out dummy assignments and get corresponding costs
    matches = []
    match_costs = []
    
    for r, c in zip(row_ind, col_ind):
        matches.append((r, c))
        match_costs.append(cost_matrix[r, c])  # Store the cost of the match
    
    return matches, match_costs


def find_nearest_sdpp_lane(av2_lanes, sdpp_lanes):
    """
    Find the nearest SD++ lane for each AV2 lane based on Chamfer distance.
    Args:
        av2_lanes: List of NumPy arrays, each of shape (N, 2).
        sdpp_lanes: List of NumPy arrays, each of shape (M, 2).
    Returns:
        min_distances: List of minimum Chamfer distances for each AV2 lane.
        nearest_matches: List of tuples (i, j), where i is the index of the AV2 lane,
                         and j is the index of the nearest SD++ lane.
    """
    cost_matrix = compute_cost_matrix(av2_lanes, sdpp_lanes)
    
    min_distances = []
    nearest_matches = []
    
    for i in range(len(av2_lanes)):
        min_idx = np.argmin(cost_matrix[i])  # Find the SD++ lane with min Chamfer distance
        min_distances.append(cost_matrix[i, min_idx])
        nearest_matches.append((i, min_idx))
    
    return min_distances, nearest_matches


def interpolate_lane(lane, num_points):
    """
    Interpolates a lane to match the desired number of points while:
    - Keeping all original points
    - Interpolating new points proportionally to segment lengths

    Args:
        lane (np.ndarray): (k, 2) array representing the original lane (k points).
        num_points (int): Target number of points (n), where n > k.

    Returns:
        np.ndarray: (n, 2) array with interpolated lane points.
    """
    k = len(lane)
    if k >= num_points:
        return lane  # No interpolation needed if we already have enough points

    # Compute segment lengths
    segment_lengths = np.linalg.norm(np.diff(lane, axis=0), axis=1)
    total_length = np.sum(segment_lengths)

    # Determine how many points to allocate per segment proportionally
    num_new_points = num_points - k  # Number of additional points needed
    segment_ratios = segment_lengths / total_length  # Proportion of total length
    segment_new_counts = np.round(segment_ratios * num_new_points).astype(int)

    # Adjust total count to exactly match num_points
    difference = num_new_points - np.sum(segment_new_counts)
    if difference > 0:  # Distribute remaining points to longest segments
        for _ in range(difference):
            longest_segment = np.argmax(segment_lengths)
            segment_new_counts[longest_segment] += 1
    elif difference < 0:  # Remove excess points from longest segments
        for _ in range(-difference):
            longest_segment = np.argmax(segment_new_counts)
            segment_new_counts[longest_segment] -= 1

    # Generate new interpolated points per segment
    interpolated_points = [lane[0]]  # Start with the first original point

    for i in range(k - 1):
        p1, p2 = lane[i], lane[i + 1]
        num_interp = segment_new_counts[i]

        if num_interp > 0:
            t_values = np.linspace(0, 1, num_interp + 2)[1:-1]  # Avoid endpoints
            new_points = (1 - t_values[:, None]) * p1 + t_values[:, None] * p2
            interpolated_points.extend(new_points)

        interpolated_points.append(p2)  # Add next original point

    return np.array(interpolated_points)


def get_lane_centerlines(file_path, road_index): # SD++
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if str(road_index) not in data:
        return None
    
    road_data = data[str(road_index)]
    lanes = road_data.get("lanes", [])
    
    lane_centerlines = [np.array(lane) for lane in lanes]
    
    return lane_centerlines


def get_road_boundaries(file_path, road_index): # SD++
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if str(road_index) not in data:
        raise ValueError("Invalid road index provided.")
    
    road_data = data[str(road_index)]
    if "left_bike_lane" in road_data:
        left_boundary = np.array(road_data.get("left_bike_lane", []))
    else:
        left_boundary = np.array(road_data.get("left_road_boundary", []))

    if "right_bike_lane" in road_data:
        right_boundary = np.array(road_data.get("right_bike_lane", []))
    else:
        right_boundary = np.array(road_data.get("right_road_boundary", []))
    
    return left_boundary, right_boundary


def get_lane_boundaries(data, lane_id): # AV2
    lane_segments = data.get("lane_segments", {})
    lane_info = lane_segments.get(str(lane_id))
    
    if lane_info:
        left_boundary = lane_info.get("left_lane_boundary", [])
        right_boundary = lane_info.get("right_lane_boundary", [])
        
        left_boundary_array = np.array([[point["x"], point["y"], point["z"]] for point in left_boundary])
        right_boundary_array = np.array([[point["x"], point["y"], point["z"]] for point in right_boundary])

    return left_boundary_array[:, :2], right_boundary_array[:, :2]


# Coordinate Transformations - SD++ to OSM
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

def convert_ab_to_ll_np(lane: np.ndarray, minlat, maxlat, minlon, maxlon):
    X, Y = lane[:, 0], lane[:, 1]
    h = haversine(minlon, minlat, minlon, maxlat)*1e7
    w = haversine(minlon, minlat, maxlon, minlat)*1e7
    lon = (X/w)*(maxlon-minlon)+minlon
    lat = (h-Y)/h*(maxlat-minlat)+minlat
    return np.stack((lat, lon), axis=1)

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

# ----------------------------------------------------------------------------------------------
# Build OSM (Road) - AV2 (Lane) Dict

city_origins = utm.CITY_ORIGIN_LATLONG_DICT
with open('osm_maps/city_utm.json', 'r') as f:
    city_origin_dict = json.load(f)
city_name_dict = {
    'austin': 'ATX', 'dearborn': 'DTW', 'miami':'MIA', 'washington-dc':'WDC',
    'pittsburgh':'PIT', 'palo-alto':'PAO'
}


def parse_osm_ways(osm_file):
    """
    Parses the OSM XML file and extracts way geometries.
    Returns a dictionary with way IDs as keys and LineString geometries as values.
    """
    root = ET.parse(osm_file)
    # root = tree.getroot()

    # Dictionary to store node coordinates
    nodes = {}
    for node in root.findall("node"):
        node_id = node.attrib["id"]
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        nodes[node_id] = (lon, lat)  # Store as (longitude, latitude)

    # Dictionary to store way geometries
    ways = {}
    for way in root.findall("way"):
        way_id = way.attrib["id"]
        nds = [nd.attrib["ref"] for nd in way.findall("nd")]

        # Convert node references to coordinates
        way_coords = [nodes[nd] for nd in nds if nd in nodes]
        if len(way_coords) > 1:  # Ensure it's a valid way
            ways[way_id] = LineString(way_coords)

    return ways


def find_closest_way(osm_file, lat, lon):
    """
    Finds the closest OSM way ID to the given latitude and longitude.
    """
    ways = parse_osm_ways(osm_file)
    if not ways:
        return "No ways found in the OSM file."

    point = Point(lon, lat)
    closest_way_id = None
    min_distance = float('inf')

    for way_id, way_geom in ways.items():
        distance = way_geom.distance(point)
        if distance < min_distance:
            min_distance = distance
            closest_way_id = way_id

    return closest_way_id if closest_way_id else "No close way found."


def gen_results(filename, llm_type, llm_model):
    map_path = f'osm_maps/{filename}/log_map_archive_{filename}.json'
    scenario_path = f'osm_maps/{filename}/scenario_{filename}.parquet'
    osm_path = f'osm_maps/{filename}/scenario_{filename}_r_200.osm'
    s_df = pd.read_parquet(scenario_path, engine='fastparquet')
    city_name = s_df.loc[0]['city']
    city_name_code = city_name_dict[s_df.loc[0]['city']]
    am = ArgoverseStaticMap.from_json(Path(map_path))
    lane_ids = am.get_scenario_lane_segment_ids()

    filtered_osm_list = filtered_way_ids(f'outputs/av2/{filename}_no_buildings_footways_pedestrian_garden_natural_landuse_author_relations.osm')

    json_file_path = f'osm_maps/{filename}_lane2osm.json'

    with open(map_path, 'r') as f:
        map_data = json.load(f)

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            lane2osm = json.load(f)
    else:
        lane2osm = defaultdict(list)

        # for lane_id in tqdm(lane_ids):
        for lane_id in lane_ids:
            lane_count = {}
            centerlines = am.get_lane_segment_centerline(lane_id)
            
            for pt in centerlines:
                latlon = utm.convert_city_coords_to_wgs84(pt[:2].reshape((1,-1)), city_name_code)
                way_id = find_closest_way(osm_path, latlon[0,0], latlon[0,1])
                if way_id not in lane_count.keys():
                    lane_count[way_id] = 0
                lane_count[way_id] += 1
            osm_out = max(lane_count, key=lane_count.get)
            lane2osm[osm_out].append(lane_id) 

        lane2osm = {key: value for key, value in lane2osm.items() if key in filtered_osm_list}
        lane2osm = {key: value for key, value in lane2osm.items() if is_not_intersection(map_data, key)}

        # Save the lane2osm dictionary to the JSON file
        with open(json_file_path, 'w') as f:
            json.dump(lane2osm, f, indent=4)


# ----------------------------------------------------------------------------------------------
# Build SD++ (Road) - OSM (Road) Dict

    abs_path = f'./outputs/av2/{filename}_output.json'

    # Load the JSON file
    with open(abs_path, 'r') as file:
        data = json.load(file)

    # Extract roads data
    roads = data.get("streets", {}).get("roads", [])

    # Create a dictionary with OSM IDs as keys and road IDs as values
    osm_to_road_ids = defaultdict(list)

    # for road in tqdm(roads):
    for road in roads:
        road_id = road[1]['id']
        osm_ids = road[1].get('osm_ids', [])       
        for osm_id in osm_ids:
            osm_to_road_ids[osm_id].append(road_id)

# ----------------------------------------------------------------------------------------------
# Get SD++ and AV2 Lanes
    min_lat, max_lat, min_lon, max_lon = get_osm_bounds(osm_path)
    av2_set = set()
    av2_matched_set = set()

    file_path = f"outputs/av2/{folder}{filename}_{llm_type}_{llm_model}_hd_output.json"

    chamfer_global = []

    for osm_id, av2_lane_ids in lane2osm.items():
        av2_lane_centerlines = []
        av2_lane_left_boundaries = []
        av2_lane_right_boundaries = []

        for lane_id in av2_lane_ids:
            # Get centerline (2D)
            centerline = am.get_lane_segment_centerline(lane_id)[:, :2]
            av2_lane_centerlines.append(centerline)

            # Get left and right boundaries (2D)
            left_boundary = am.vector_lane_segments[lane_id].left_lane_boundary.xyz[:, :2]
            right_boundary = am.vector_lane_segments[lane_id].right_lane_boundary.xyz[:, :2]

            av2_lane_left_boundaries.append(left_boundary)
            av2_lane_right_boundaries.append(right_boundary)       

        for id_int in av2_lane_ids:
            av2_set.add(id_int)

        sd_lane_centerlines = []

        for road_id in osm_to_road_ids[int(osm_id)]:
            lane_boundaries = get_lane_centerlines(file_path, road_id)
            if lane_boundaries is None:
                continue
            
            left_boundary, right_boundary = get_road_boundaries(file_path, road_id)
            
            if len(lane_boundaries) == 0:
                sd_lane_centerlines.append((left_boundary + right_boundary) / 2)
            else:
                sd_lane_centerlines.append((left_boundary + lane_boundaries[0]) / 2)
                for i in range(len(lane_boundaries) - 1):
                    sd_lane_centerlines.append((lane_boundaries[i] + lane_boundaries[i + 1]) / 2)
                sd_lane_centerlines.append((lane_boundaries[-1] + right_boundary) / 2)

        sd_lane_centerlines_ll = [convert_ab_to_ll_np(centerline, min_lat, max_lat, min_lon, max_lon)
                               for centerline in sd_lane_centerlines]
        
        city_code = city_name_dict[city_name.lower()]  # Convert city name to lowercase to avoid case mismatches
        city_enum = CityName[city_code]  # Convert city code to CityName Enum

        sd_lane_centerlines_utm = [
            np.array([
                convert_gps_to_utm(lat, lon, city_enum)  # Apply function to each row
                for lat, lon in centerline
            ])
            for centerline in sd_lane_centerlines_ll
        ]

        sd_lane_centerlines_city = [
            convert_utm_to_city_coords(centerline_utm, city_enum)  # Apply function to each 2D array
            for centerline_utm in sd_lane_centerlines_utm
        ]

        min_x, max_x, min_y, max_y = get_hdmap_bounds(map_data)

        sd_lane_centerlines_city_filtered = filter_lanes_sdpp(sd_lane_centerlines_city, min_x, max_x, min_y, max_y)

        sd_lane_centerlines_utm_filtered = [
            convert_city_coords_to_utm(centerline_utm, city_enum)  # Apply function to each 2D array
            for centerline_utm in sd_lane_centerlines_city_filtered
        ]    

        if len(sd_lane_centerlines_utm_filtered) == 0:
            continue

        min_dist, matches = find_nearest_sdpp_lane(av2_lane_centerlines, sd_lane_centerlines_city_filtered)

        for i in range(len(min_dist)):
            if min_dist[i] <= 5:
                av2_matched_set.add(av2_lane_ids[matches[i][0]])

        chamfer_global.extend(min_dist)

    return (av2_set, av2_matched_set, chamfer_global)


def main(llm_type, llm_model, map_name):
    av2_set, av2_matched_set, chamfer = gen_results(map_name, llm_type, llm_model)

    if len(chamfer):
        print("Average Chamfer Distance:", sum(chamfer) / len(chamfer))
        print("Max Chamfer", max(chamfer))
        print("Min Chamfer", min(chamfer))
        std_dev = statistics.stdev(chamfer)
        print("Standard Deviation:", std_dev)
    else:
        print('No Matches')

    print('Recall', av2_matched_set / av2_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LLM and Map details.")
    parser.add_argument("llm_type", type=str, help="Type of the LLM (e.g., openai, llama)")
    parser.add_argument("llm_model", type=str, help="Exact model name of the LLM (e.g., gpt-4o, llama3.3:70b)")
    parser.add_argument("map_name", type=str, help="Name or ID of the map")

    args = parser.parse_args()

    main(args.llm_type, args.llm_model, args.map_name)