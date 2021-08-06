import pandas as pd
from glob import glob
from functools import reduce
from sklearn.preprocessing import LabelEncoder
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os.path import join

description = """
This script creates a matrix of presence/absence for each route in the BBS dataset.
For help about the arguments, call python extract_route_pa.py --help
If all goes well, it should produce two files:
1) route_pa_{year}.csv: The table of presence/absence by route
2) species_info_{year}.csv: Information about each species
Note that a species is marked as present if it is observed at least once on
_any_ of the 50 stops, so the spatial resolution is quite coarse.
"""

parser = ArgumentParser(
    description=description, formatter_class=RawDescriptionHelpFormatter
)
parser.add_argument(
    "--base-dir",
    required=True,
    type=str,
    help="The path to the 50-StopData. For example: ./50-StopData/1997ToPresent_SurveyWide",
)

parser.add_argument(
    "--n-stops",
    required=False,
    type=int,
    default=50,
    help="If specified, only the first n stops are used to create"
    "presence/absence. This can be helpful to increase precision of the"
    "coordinates, since they are only given at the start of the route.",
)

parser.add_argument(
    "--year",
    required=False,
    default=2019,
    type=int,
    help="The year to extract route PA for. Defaults to 2019.",
)
parser.add_argument(
    "--min-observations",
    required=False,
    default=10,
    type=int,
    help="The minimum number of presences for a bird species to be kept. Defaults to 10.",
)

args = parser.parse_args()

year = args.year
min_observations = args.min_observations
base_dir = args.base_dir

stop_files = glob(join(base_dir, "fifty*.csv"))
stop_data = pd.concat([pd.read_csv(x) for x in stop_files])
routes_data = pd.read_csv("../routes.csv", encoding="latin-1")

stop_data = stop_data[stop_data["Year"] == year]

# Route numbers are _not_ unique. The same route can be in a different country
# and state!
stop_cols = [x for x in stop_data.columns if "Stop" in x]
stop_obs = stop_data[stop_cols]

n_stops = args.n_stops
stops_to_fetch = list(range(1, n_stops + 1))
stop_obs = stop_obs[[f"Stop{n}" for n in stops_to_fetch]]

stop_data["was_observed"] = stop_obs.sum(axis=1) > 0
stop_data = stop_data[stop_data["was_observed"]]

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


def split_line(line):

    stripped = line.strip()

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

    fields = [
        info_line[x + 1 : y + 1].strip()
        for x, y in zip(split_indices, split_indices[1:])
    ]
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
relevant = relevant.rename(columns={"English_Common_Name": "common_name"})
with_species_info = with_coords.merge(relevant, on="AOU")

# The next step is to turn this into a presence/absence matrix by route.
species_encoder = LabelEncoder()
species_encoder.fit(with_species_info["scientific_name"])

route_encoder = LabelEncoder()
route_encoder.fit(with_species_info["route_id"])

n_routes = len(route_encoder.classes_)
n_species = len(species_encoder.classes_)

route_pa_mat = np.zeros((n_routes, n_species), dtype=int)

route_ids = route_encoder.transform(with_species_info["route_id"])
sp_ids = species_encoder.transform(with_species_info["scientific_name"])

combinations = pd.Series(route_ids.astype(str)) + "-" + pd.Series(sp_ids.astype(str))

# Make sure these are unique
assert np.max(combinations.value_counts() == 1)

# Fill the matrix
route_pa_mat[route_ids, sp_ids] = 1

pa_df = pd.DataFrame(
    route_pa_mat, index=route_encoder.classes_, columns=species_encoder.classes_
)

# We also need the coordinates for each route
def get_coordinates(route_df):

    # Make sure they are unique:
    assert len(route_df["Latitude"].unique()) == 1
    assert len(route_df["Longitude"].unique()) == 1

    return pd.Series(
        {
            "Latitude": route_df["Latitude"].iloc[0],
            "Longitude": route_df["Longitude"].iloc[0],
        }
    )


route_coords = with_species_info.groupby("route_id").apply(get_coordinates)

pa_df = pd.concat([pa_df, route_coords], axis=1)

if args.n_stops < 50:
    suffix = f"up_to_{n_stops}_only"
else:
    suffix = ""

# Store the results
pa_df.to_csv(f"route_pa_{year}_{suffix}.csv")
relevant.to_csv(f"species_info_{year}_{suffix}.csv")
