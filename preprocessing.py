import os
import argparse

def remove_pos_tagging(sequences: list) -> list:
    for sequence in sequences:
        yield sequence.replace('_nn', '').replace('_vb', '')

def split_sequences(sequences:list, max_len:int=100) -> list:
    for sequence in sequences:
        if max_len == 0:
            yield sequence
        for split in split_sequence(sequence, max_len):
            yield " ".join(split)

def split_sequence(text:str, max_len:int=100)-> list:
        if len(text) <= max_len:
            yield text.split()
        else:
            count = 0
            sequence_jr = list()

            for token in text.split():
                count += len(token) + 1  # blanckspace
                sequence_jr.append(token)
                if count >= max_len:
                    sequence = text[count:]
                    break

            yield sequence_jr

            for split in split_sequence(sequence, max_len):
                yield split

def remove_numbers(sequences:list) -> list:
    for sequence in sequences:
        yield "".join([token for token in sequence if not token.isdecimal()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_paths", default='data/corpora/ccoha1.txt;data/corpora/ccoha2.txt',
                        type=str,
                        help="Paths to all corpus time slices separated by ';'.")
    parser.add_argument("--output_corpus_paths", default='data/corpora/processed_ccoha1.txt;data/corpora/processed_ccoha2.txt', type=str,
                        help="Paths to all processed corpus time slices separated by ';'.")
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin'])
    args = parser.parse_args()


    corpus_paths = args.corpus_paths.split(';')
    output_corpus_paths = args.output_corpus_paths.split(';')
    for i, corpus_path in enumerate(corpus_paths):
        lines = open(corpus_path, mode='r', encoding='utf-8').read().split("\n")

        # remove artifacts from semeval '_vb', '_nn'
        lines = remove_pos_tagging(lines)

        # split long sequences (more than 500 characters long)
        if args.language == 'latin':
            lines = split_sequences(lines, max_len=500)

        # remove numbers
        lines = remove_numbers(lines)

        # save processing
        os.makedirs(os.path.dirname(output_corpus_paths[i]), exist_ok=True)
        open(output_corpus_paths[i], mode='w', encoding='utf-8').write('\n'.join(lines))