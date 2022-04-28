from pathlib import Path

def transform_data(input_path, output_path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    trans_lines = []
    with open(input_path, 'r', encoding='utf-8') as inputfd:
        lines = inputfd.readlines()
        for line in lines:
            tokens = line.split(' ')
            trans_tokens = []
            for token in tokens:
                tags = token.split('SEPL|||SEPR')
                labels = tags[-1].split('SEPL__SEPR')
                trans_labels = []
                for label in labels:
                    trans_labels.append(label.split('_')[0])
                tags[-1] = 'SEPL__SEPR'.join(trans_labels)
                trans_tokens.append('SEPL|||SEPR'.join(tags))
            trans_lines.append(' '.join(trans_tokens))
    with open(output_path, 'w', encoding='utf-8') as outputfd:
        outputfd.writelines(trans_lines)  
    print(f'Successfully transform {input_path} to {output_path}')        

if __name__ == '__main__':
    train_file = Path('./CGEDprocessed/train.txt')
    dev_file = Path('./CGEDprocessed/dev.txt')

    transform_data(train_file, './CGEDprocessed/d_train.txt')
    transform_data(dev_file, './CGEDprocessed/d_dev.txt')
