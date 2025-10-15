#!/bin/bash
# Usage: ./preprocess_osm_av2.sh <map_path>
# map_path would look like "..Argoverse2/motion/val/d9910975-9eaa-48df-92b7-c72ac01c17a3/scenario_d9910975-9eaa-48df-92b7-c72ac01c17a3_r_200.osm"

start=$SECONDS

map_path=$1
map_name=(${map_path//\// })
map_name=${map_name[-2]}    # e.g. d9910975-9eaa-48df-92b7-c72ac01c17a3

output_path="outputs/av2/"

osmium tags-filter -i $1 /building /highway=footway /highway=pedestrian /garden:type /natural /landuse --omit-referenced -o "$output_path""$map_name"_no_buildings_footways_pedestrian_garden_natural_landuse.osm --overwrite

osmfilter "$output_path""$map_name"_no_buildings_footways_pedestrian_garden_natural_landuse.osm --drop-author --drop-relations -o="$output_path""$map_name"_no_buildings_footways_pedestrian_garden_natural_landuse_author_relations.osm 

cd ./abstreet && ./binaries/cli oneshot-import ../"$output_path""$map_name"_no_buildings_footways_pedestrian_garden_natural_landuse_author_relations.osm

./binaries/cli dump-json ./data/input/zz/oneshot/raw_maps/"$map_name"_no_buildings_footways_pedestrian_garden_natural_landuse_author_relations.bin > ../"$output_path""$map_name"_output.json

duration=$(( SECONDS - start ))
echo "Time taken for preprocessing OSM: $duration seconds"