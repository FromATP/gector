from pathlib import Path
from helpers import SEQ_DELIMETERS

def get_label_dict(label_file):
    with open(label_file, "r", encoding="utf-8") as inputfd:
        lines = inputfd.readlines()
    output_dict = {}
    for line in lines:
        src_sent = []
        label_list = []
        tokens = line[:-1].split(SEQ_DELIMETERS["tokens"])
        for token in tokens:
            word, tags = token.split(SEQ_DELIMETERS["labels"])
            tags = tags.split(SEQ_DELIMETERS["operations"])
            cur_tag = tags[0].split("_")[0]
            src_sent.append(word)
            label_list.append(cur_tag)
        src_sent = "".join(src_sent[1:])
        label_list = label_list[1:]
        output_dict[src_sent] = label_list

    print(f"Totally {len(output_dict)} sents.")
    return output_dict

def get_GED_truth(label_dict, inputfile, outputfile):
    with open(inputfile, "r", encoding="utf-8") as inputfd:
        lines = inputfd.readlines()
    data = []
    for line in lines:
        src, tgt = line[:-1].split(SEQ_DELIMETERS["sents"])
        src_key = "".join(src.split(SEQ_DELIMETERS["tokens"])[1:-1])
        label = ["$KEEP"] + label_dict[src_key] + ["$KEEP"]
        assert len(label) == len(src.split(SEQ_DELIMETERS["tokens"]))
        label = SEQ_DELIMETERS["tokens"].join(label)
        data.append(SEQ_DELIMETERS["sents"].join([src, tgt, label]))
    with open(outputfile, "w", encoding="utf-8") as outputfd:
        for line in data:
            outputfd.write(line + "\n")
    print(f"{len(data)} in", str(outputfile))

if __name__ == "__main__":
    data_dir = Path("./CGEDprocessed")
    train_file = data_dir / "gec_train.txt"
    dev_file = data_dir / "gec_dev.txt"
    label_file = data_dir / "labelled.txt"

    label_dict = get_label_dict(label_file)

    get_GED_truth(label_dict, train_file, data_dir / "truth_train.txt")
    get_GED_truth(label_dict, dev_file, data_dir / "truth_dev.txt")

    print("Finished.")