import os
import sys

def convert_sms_spam_collection(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    index_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip() == "":
            continue  # Skip empty lines
        try:
            label, text = line.strip().split('\t', 1)  # 尝试用Tab分
        except ValueError:
            label, text = line.strip().split(' ', 1)  # 如果不行，再用空格分
            label = label.strip()
            text = text.strip()

        filename = f"{i:05d}.txt"  # 文件名格式：00000.txt，00001.txt
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.write(text)

        index_lines.append(f"{label} {filename}")

    # 写 index 文件
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