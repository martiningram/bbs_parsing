import pandas as pd
from glob import glob
from functools import reduce
from sklearn.preprocessing import LabelEncoder
import numpy as np


year = 2019
min_observations = 10

stop_files = glob("../50-StopData/1997ToPresent_SurveyWide/fifty*.csv")
stop_data = pd.concat([pd.read_csv(x) for x in stop_files])
routes_data = pd.read_csv("../routes.csv", encoding="latin-1")

stop_data = stop_data[stop_data["Year"] == year]

# We only care about presence-absence at the level of routes. So do sums of all
# the stops.
# Route numbers are _not_ unique. The same route can be in a different country
# and state!
stop_cols = [x for x in stop_data.columns if "Stop" in x]
stop_obs = stop_data[stop_cols]
stop_data["was_observed"] = stop_obs.sum(axis=1) > 0

assert all(stop_data["was_observed"].values)

stop_data = stop_data.drop(columns=stop_cols)
with_coords = stop_data.merge(routes_data)

# Drop inactive routes
with_coords = with_coords[with_coords["Active"] == 1]

# Drop RouteTypeDetailID = 3, apparently not of good quality (see: https://github.com/martiningram/mistnet/blob/0e120b814309679f07590426757778bb7f5f8bd7/extras/BBS-analysis/data_extraction/data-extraction.R)
with_coords = with_coords[with_coords["RouteTypeDetailID"].isin([1, 2])]

species_counts = with_coords["AOU"].value_counts()
to_keep = species_counts[species_counts > min_observations].index

with_coords = with_coords[with_coords["AOU"].isin(to_keep)]

route_id_cols = ["RouteDataID", "CountryNum", "StateNum", "Route"]
route_ids = with_coords[route_id_cols].astype(str)
route_ids = reduce(
    lambda x, y: x + "-" + y if x is not None else y, route_ids.values.T, None
)
with_coords["route_id"] = route_ids

# Next: Replace the AOU with a species name; maybe keep both common name and
# scientific name
species_list_file = "../SpeciesList.txt"

# We need to read this one line-by-line
all_lines = list(open(species_list_file, encoding="latin-1"))


def get_header(header_line):

    stripped = header_line.strip()

    return [x for x in stripped.split(" ") if x != ""]


# Read header
header_line = all_lines[9]
split_header = split_line(header_line)

# Make sure this worked
assert "AOU" in split_header

# This line gives the maximum length of things to parse
length_line = all_lines[10]
split_lengths = split_line(length_line)
split_lengths = [len(x) for x in split_lengths]
split_indices = [0] + list(np.cumsum(split_lengths))


def split_info(info_line, split_indices, header):

    fields = [info_line[x:y].strip() for x, y in zip(split_indices, split_indices[1:])]
    with_names = {x: y for x, y in zip(header, fields)}

    return with_names


all_split = pd.DataFrame(
    [
        split_info(sample_line, split_indices, split_header)
        for sample_line in all_lines[11:]
    ]
)

all_split["AOU"] = all_split["AOU"].astype(int)
all_split["scientific_name"] = all_split["Genus"] + " " + all_split["Species"]

relevant = all_split[["AOU", "scientific_name", "English_Common_Name"]]

with_species_info = with_coords.merge(relevant, on="AOU")
