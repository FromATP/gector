from pathlib import Path

if __name__ == "__main__":
    data_dir = Path('./CGEDrawdata')
    output_dir = Path('./CGEDprocessed')
    inputfile = data_dir / '2021' / 'test_2021.txt'
    outputfile = output_dir / 'test.txt'
    sent = []

    with open(inputfile, 'r', encoding='utf-8') as inputfd:
        lines = inputfd.readlines()
        for line in lines:
            line = line.split('\t')
            line = ' '.join(list(line[1]))
            sent.append(line)
    print(f'{len(sent)} test sentences in total.')
    with open(outputfile, 'w', encoding='utf-8') as outputfd:
        outputfd.writelines(sent)