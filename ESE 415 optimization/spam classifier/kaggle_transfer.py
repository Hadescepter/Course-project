import os
import sys
import csv

def convert_sms_spam_collection(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    index_lines = []

    # 注意这里改成 ISO-8859-1
    with open(input_file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)  # 跳过标题行

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            label = row[0].strip().lower()
            text = row[1].strip()

            filename = f"{i:05d}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as out_f:
                out_f.write(text)

            index_lines.append(f"{label} {filename}")

    with open(os.path.join(output_dir, 'index'), 'w', encoding='utf-8') as idx_f:
        idx_f.write('\n'.join(index_lines))

    print(f"Finished converting {len(index_lines)} messages into {output_dir}/")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_sms_to_index_format.py <sms_spam_collection_file> <output_directory>")
    else:
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        convert_sms_spam_collection(input_file, output_dir)