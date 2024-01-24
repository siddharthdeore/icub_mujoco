# python scipt to apply meshlab filters to all stl files in a directory

import os
import subprocess
import argparse
import pymeshlab as ml
ml.print_filter_list()
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./stl", help="input directory")
parser.add_argument('--output_dir', type=str, default="./stl_bin", help="output directory")

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = os.listdir(input_dir)
files = [f for f in files if f.endswith(".stl")]

for f in files:
    print("processing {}".format(f))
    input_path = os.path.join(input_dir, f)
    output_path = os.path.join(output_dir, f)
    ms = ml.MeshSet()
    
    ms.load_new_mesh(input_path)
    ms. meshing_decimation_quadric_edge_collapse()
    ms.save_current_mesh(output_path)


    