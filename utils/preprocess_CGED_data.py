from cmath import inf
from pathlib import Path
from xml.etree import ElementTree
from sklearn.model_selection import train_test_split
from zhconv import convert

from Levenshtein import distance

from preprocess_data import convert_data_from_raw_files

def check_consistency(src_dir, tgt_dir, output_dir):
    src_lines = []
    tgt_lines = []
    with open(src_dir, 'r', encoding='utf-8') as srcfd:
        src_lines = srcfd.readlines()
    with open(tgt_dir, 'r', encoding='utf-8') as tgtfd:
        tgt_lines = tgtfd.readlines()

    if len(src_lines) != len(tgt_lines):
        print(f'WARNING: The source and target are not the same length.')
        print(f'Source file: {src_dir}, size {len(src_lines)}')
        print(f'Target file: {tgt_dir}, size {len(tgt_lines)}')
    else:
        src_file = output_dir / 'all_sent.src'
        tgt_file = output_dir / 'all_sent.tgt'
        count = 0 
        with open(src_file, 'a', encoding = 'utf-8') as srcfd, open(tgt_file, 'a', encoding = 'utf-8') as tgtfd:
            for src, tgt in zip(src_lines, tgt_lines):
                srcfd.write(' '.join(src.strip()) + '\n')
                tgtfd.write(' '.join(tgt.strip()) + '\n')
                count += 1
        return count

    md_src = [(0, inf) for i in src_lines]
    md_tgt = [(0, inf) for i in tgt_lines]
    for i, src in enumerate(src_lines):
        for j, tgt in enumerate(tgt_lines):
            cur = distance(src, tgt)
            if cur < md_src[i][1]:
                md_src[i] = (j, cur)
            if cur < md_tgt[j][1]:
                md_tgt[j] = (i, cur)
    
    src_file = output_dir / 'all_sent.src'
    tgt_file = output_dir / 'all_sent.tgt'
    count = 0 
    with open(src_file, 'a', encoding = 'utf-8') as srcfd, open(tgt_file, 'a', encoding = 'utf-8') as tgtfd:
        for i, pair in enumerate(md_src):
            if md_tgt[pair[0]][0] == i:
                srcfd.write(' '.join(src_lines[i].strip()) + '\n')
                tgtfd.write(' '.join(tgt_lines[pair[0]].strip()) + '\n')
                count += 1
    return count

def convert_to_src_tgt_form(src_dir, output_dir):
    tree = ElementTree.parse(src_dir)
    root = tree.getroot()
    src_file = output_dir / 'all_sent.src'
    tgt_file = output_dir / 'all_sent.tgt'
    count = 0 

    with open(src_file, 'a', encoding = 'utf-8') as srcfd, open(tgt_file, 'a', encoding = 'utf-8') as tgtfd:
        for doc in root:
            for attr in doc:
                if attr.tag.upper() == 'TEXT':
                    tmp = convert(attr.text, 'zh-cn') 
                    srcfd.write(' '.join(tmp.strip()) + '\n')
                if attr.tag.upper() == 'CORRECTION':
                    tmp = convert(attr.text, 'zh-cn')
                    tgtfd.write(' '.join(tmp.strip()) + '\n')
            count += 1
    return count

def process2014(input_dir, output_dir):
    count = 0
    count += check_consistency(input_dir / '14_test.src', input_dir / '14_test.tgt', output_dir)
    count += check_consistency(input_dir / '14_train.src', input_dir / '14_train.tgt', output_dir)
    return count

def process2015(input_dir, output_dir):
    count = 0
    count += check_consistency(input_dir / '15_test.src', input_dir / '15_test.tgt', output_dir)
    count += check_consistency(input_dir / '15_train.src', input_dir / '15_train.tgt', output_dir)
    return count

def process2016(input_dir, output_dir):
    count = 0
    count += convert_to_src_tgt_form(input_dir / 'CGED16_HSK_TrainingSet.txt', output_dir)
    count += convert_to_src_tgt_form(input_dir / 'CGED16_TOCFL_TrainingSet.txt', output_dir)
    return count
 
def process2017(input_dir, output_dir):
    count = 0
    check_consistency(input_dir / '2017_train.src', input_dir / '2017_train.tgt', output_dir)
    return count

def process2018(input_dir, output_dir):
    count = 0
    count += check_consistency(input_dir / '2018_train.src', input_dir / '2018_train.tgt', output_dir)
    return count

def process2020(input_dir, output_dir):
    count = 0
    count += convert_to_src_tgt_form(input_dir / 'CGED2020train.xml', output_dir)
    return count

def split_train_test(inputfile, trainfile, testfile):
    train = []
    test = []
    with open(inputfile, 'r', encoding='utf-8') as inputfd:
        text = inputfd.readlines()
        train, test = train_test_split(text, test_size = 0.15)
    with open(trainfile, 'w', encoding='utf-8') as outputfd:
        for line in train:
            outputfd.write(line)
    with open(testfile, 'w', encoding='utf-8') as outputfd:
        for line in test:
            outputfd.write(line)
    print(f'{len(train)} sentences in training set and {len(test)} sentences in development set.')

if __name__ == '__main__':
    data_dir = Path('./CGEDrawdata')
    output_dir = Path('./CGEDprocessed')
    output_dir.mkdir(exist_ok=True, parents=True)
    count = 0
    count += process2014(data_dir / '2014', output_dir)
    count += process2015(data_dir / '2015', output_dir)
    count += process2016(data_dir / '2016', output_dir)
    count += process2017(data_dir / '2017', output_dir)
    count += process2018(data_dir / '2018', output_dir)
    count += process2020(data_dir / '2020', output_dir)
    # 2021: only test set
    print(f'{count} parallel sentences in total.')

    convert_data_from_raw_files(
        output_dir / 'all_sent.src', output_dir / 'all_sent.tgt', output_dir / 'labelled.txt', 1000000)

    split_train_test(output_dir / 'labelled.txt', output_dir / 'train.txt', output_dir / 'dev.txt')