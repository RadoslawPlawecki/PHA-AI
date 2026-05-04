import os

input_root = "data/phanotate/proteins"
output_root = "data/phanotate/merged"

os.makedirs(output_root, exist_ok=True)

tool_to_output = {
    "genomad": "geN_ChV_PHA_PROT.faa",
    "vibrant": "VIB_ChV_PHA_PROT.faa",
    "virsorter2": "VS2_ChV_PHA_PROT.faa",
}


def merge_files(tool_dir, output_file):
    with open(output_file, "w") as outfile:
        for fname in sorted(os.listdir(tool_dir)):
            fpath = os.path.join(tool_dir, fname)

            if not os.path.isfile(fpath):
                continue

            with open(fpath) as infile:
                for line in infile:
                    outfile.write(line.rstrip() + "\n")


def main():
    for tool, out_name in tool_to_output.items():
        tool_dir = os.path.join(input_root, tool)

        if not os.path.isdir(tool_dir):
            print(f"Skipping missing directory: {tool_dir}")
            continue

        output_file = os.path.join(output_root, out_name)

        merge_files(tool_dir, output_file)
        print(f"Written: {output_file}")


if __name__ == "__main__":
    main()