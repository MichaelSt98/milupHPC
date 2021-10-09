import csv
import argparse
import os

def csv2md(csvFile, delimiter=";"):

    output_file = csvFile.replace(".csv", ".md")
    csv_dict = csv.DictReader(open(csvFile, encoding="UTF-8"), delimiter=delimiter)
    list_of_rows = [dict_row for dict_row in csv_dict]

    headers = list(list_of_rows[0].keys())

    md_string = "| "
    for header in headers:
        md_string += "**" + header + "** |"

    md_string += "\n|"
    for i in range(len(headers)):
        md_string += "--- | "

    md_string += "\n|"
    for row in list_of_rows:
        for header in headers:
            md_string += row[header]+" | "
        md_string += "\n"

    file = open(output_file, "w", encoding="UTF-8")
    file.write(md_string)
    file.close()

    print("Converted table to Markdown and saved in {}!".format(output_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert csv table to Markdown table")

    parser.add_argument('filename', type=str, help='file name')
    parser.add_argument('--delimiter', type=str, nargs='?', const=";", default=";", help='CSV file delimiter')

    args = parser.parse_args()

    if os.path.isfile(args.filename):
        csv2md(args.filename, args.delimiter)
    else:
        print("file `{}` not available!".format(args.filename))

