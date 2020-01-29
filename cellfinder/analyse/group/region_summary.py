import os
import argparse

import pandas as pd
from imlib.general.list import remove_empty_string, unique_elements_lists

from cellfinder.tools.source_files import get_structures_path


def structure_name_to_id(structure_name, reference_structures_table):
    row = reference_structures_table[
        reference_structures_table["name"] == structure_name
    ]
    return row["structure_id_path"].iloc[0]


def get_substructures(structure_name, reference_structures_table):
    structure_id = structure_name_to_id(
        structure_name, reference_structures_table
    )
    return reference_structures_table[
        reference_structures_table["structure_id_path"].str.startswith(
            structure_id
        )
    ]


def get_sub_results(
    result, structure_name, reference_structures_table, sum_regions=False
):
    substructures = get_substructures(
        structure_name, reference_structures_table
    )
    substructures_names = substructures["name"]
    sub_results = result[result["structure_name"].isin(substructures_names)]
    if sum_regions:
        sub_results = sub_results.sum().to_frame()
        sub_results = sub_results.T
        sub_results["structure_name"] = structure_name
        sub_results["left_cells_per_mm3"] = (
            sub_results["left_cell_count"] / sub_results["left_volume_mm3"]
        )
        sub_results["right_cells_per_mm3"] = (
            sub_results["right_cell_count"] / sub_results["right_volume_mm3"]
        )

    return sub_results


def make_summary_df_for_brain(
    brain_results_path, regions, reference_structures_table, sum_regions=False
):
    result = pd.read_csv(brain_results_path)
    df = pd.DataFrame()
    for region in regions:
        try:
            sub_results = get_sub_results(
                result,
                region,
                reference_structures_table,
                sum_regions=sum_regions,
            )
            df = df.append(sub_results)
        except IndexError:
            print(
                "Brain region: {} could not be found in the atlas "
                "definition. Skipping.".format(region)
            )
    return df


def summary_run(args):
    if args.structures_file_path is None:
        args.structures_file_path = get_structures_path()
        reference_structures_table = pd.read_csv(args.structures_file_path)
    else:
        raise NotImplementedError(
            "Only the Allen adult mouse atlas is " "currently supported."
        )

    if not args.regions and not args.regions_list:
        regions = list(pd.read_csv(args.structures_file_path)["name"])
    else:
        regions = []
        if args.regions:
            regions = regions + args.regions
        if args.regions_list:
            regions = regions + list(pd.read_csv(args.regions_list)["name"])

    regions = remove_empty_string(regions)
    regions = unique_elements_lists(regions)
    csvs_folder = args.csv_dir
    destination_folder = os.path.join(args.csv_dir, "summary")

    csvs_names = [f for f in os.listdir(csvs_folder) if f.endswith(".csv")]
    if len(csvs_names) == 0:
        raise FileNotFoundError(
            "No CSV files were found in the directory: "
            "{}. Please check the arguments".format(csvs_folder)
        )

    csvs_paths = [os.path.join(csvs_folder, f) for f in csvs_names]

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for csv_file_path in csvs_paths:
        print("CSV file: {}".format(os.path.basename(csv_file_path)))
        summary = make_summary_df_for_brain(
            csv_file_path,
            regions,
            reference_structures_table,
            sum_regions=args.sum_regions,
        )
        filename = os.path.basename(csv_file_path)
        dest_path = os.path.join(destination_folder, filename)

        summary.to_csv(dest_path, index=False)
    print("Done!")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="csv_dir",
        type=str,
        help="Directory containing csv files to be summarised",
    )

    parser.add_argument(
        "--regions-list",
        dest="regions_list",
        type=str,
        help="Curated structure list in a text file (as per the allen brain "
        "atlas csv)",
    )
    parser.add_argument(
        "--regions",
        dest="regions",
        nargs="+",
        help="A list of additional regions to include",
    )
    parser.add_argument(
        "--structures-file",
        dest="structures_file_path",
        type=str,
        help="The csv file containing the structures "
        "definition (if not using the default "
        "Allen brain atlas).",
    )
    parser.add_argument(
        "--sum-regions",
        dest="sum_regions",
        action="store_true",
        help="Rather than listing substructures, sum them.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    summary_run(args)


if __name__ == "__main__":
    main()
