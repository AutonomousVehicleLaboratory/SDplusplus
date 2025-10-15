import matplotlib.pyplot as plt
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--map_name', type=str)
args = parser.parse_args()

input_json = f'../outputs/{args.map_name}_hd_output.json'
output_fig = f'../outputs/{args.map_name}.png'

with open(input_json) as f:
    roads = json.load(f)

scale = 1

plt.figure(dpi=1000)

# lat lon
def convert_to_image_coords(lat, lon):
    x = ((lat) * scale)
    y = (-1*(lon) * scale)
    return x, y

def plot_polyline(polyline, linewidth=0.5, color='b', label=None):
    lats, lons = zip(*polyline)
    X_Y = [convert_to_image_coords(lat, lon) for lat, lon in zip(lats, lons)]
    X, Y = zip(*X_Y)
    plt.plot(X, Y, color, label=label,linewidth=linewidth)

def plot_points(points, color='b', label=None):
    for point in points:
        x, y = convert_to_image_coords(*point)
        plt.scatter(x, y, c=color, label=label)
        if label:
            label = None

for road_id in roads:
    # print('road id', road_id) # ADDED
    plot_polyline(roads[road_id]['centerline'], 0.5, 'm--', label='centerline')
    for lane in roads[road_id]['lanes']:
        print('Lane Detected')
        # print('lane found') # ADDED
        plot_polyline(lane, 0.4, 'b--', label='lane')

    if 'left_bike_lane' in roads[road_id]:
        print('Left Bike Lane Detected')
        plot_polyline(roads[road_id]['left_bike_lane'], 0.7, 'g-', label='bike_lane')
    if 'right_bike_lane' in roads[road_id]:
        print('Right Bike Lane Detected')
        plot_polyline(roads[road_id]['right_bike_lane'], 0.7, 'g-')

    plot_polyline(roads[road_id]['left_road_boundary'], 0.5, 'k-', label='road_boundary')
    plot_polyline(roads[road_id]['right_road_boundary'], 0.5, 'k-', )
plt.axis('off')
print('label saved') #ADDED
plt.savefig(output_fig, bbox_inches='tight', pad_inches=0)
