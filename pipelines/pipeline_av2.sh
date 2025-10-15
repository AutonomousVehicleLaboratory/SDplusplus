#!/bin/bash

# Example usage: ./pipelines/pipeline_av2.sh ./osm_maps/scenario_f5ccb9d1-5bc7-4e0c-8fa8-654197e61f53_r_200.osm openai gpt-4o

start=$SECONDS

map_path=$1
map_name=(${map_path//\// })
map_name=${map_name[-2]}

llm_type=$2
llm_model=$3

./pipelines/preprocess_osm_av2.sh $map_path
# cd ./rag && python3 rag_langchain_av2_osg_p1.py --map_name $map_name --llm_type $llm_type --llm_model $llm_model
cd ./rag && python3 rag_langchain_av2_osg_p2.py --map_name $map_name --llm_type $llm_type --llm_model $llm_model
# cd ./rag && python3 rag_langchain_av2_ig.py --map_name $map_name --llm_type $llm_type --llm_model $llm_model
# cd ./rag && python3 rag_langchain_av2_ig_context.py --map_name $map_name --llm_type $llm_type --llm_model $llm_model
cd ../codes
python3 get_HD_map_av2.py --map_name $map_name --llm_model $llm_model --llm_type $llm_type
python3 visualize_AB.py --map_name av2/$map_name"_"$llm_type"_"$llm_model

duration=$(( SECONDS - start ))
echo "Total time elapsed: $duration seconds"