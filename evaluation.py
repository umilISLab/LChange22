import os
import sys
import argparse
import pandas as pd
from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin'])
    parser.add_argument("--results_file", default="results_bert_english/results_english.csv", type=str,
                        help="Path to csv file with results")
    parser.add_argument("--gold_path", default='data/evaluation/en_graded.txt', type=str,
                        help="Path to gold data with annotator scores")
    parser.add_argument("--output_corr_file", default='results_bert_english.csv', type=str,
                        help="Path to output file with correlation score.")
    args = parser.parse_args()

    # language
    lang = args.language
    languages = ['english', 'latin']
    if lang not in languages:
        print("Language not valid, valid choices are: ", ", ".join(languages))
        sys.exit()
    print("Language:", lang.upper())

    results = list()

    df = pd.read_csv(args.results_file, sep='\t').sort_values(by=['word'])
    df_truth = pd.read_csv(args.gold_path, sep='\t', names=['word', 'score']).sort_values(by=['word'])

    baselines = ['cd', 'div']
    clustering = ['ap', 'iapna', 'app']
    measures = ['jsd', 'pdis', 'pdiv']

    for baseline in baselines:
        corr = stats.spearmanr(df[baseline].values, df_truth.score.values)
        score = corr.correlation
        pvalue = corr.pvalue
        print('\n=======', baseline.upper(), '=======')
        print('Corr:', round(score, 3), '\t\t', 'pvalue:', round(pvalue, 3))
        row = {'method': baseline, 'corr': score, 'pvalue':pvalue}
        results.append(row)


    for clus in clustering:
        for measure in measures:
            method = clus+'_'+measure
            corr = stats.spearmanr(df[method].values, df_truth.score.values)
            score = corr.correlation
            pvalue = corr.pvalue
            print('\n=======', method.upper(), '=======')
            print('Corr:', round(score, 3), '\t\t', 'pvalue:', round(pvalue, 3))
            row = {'method': method, 'corr': score, 'pvalue': pvalue}
            results.append(row)

    dir = os.path.dirname(args.output_corr_file)
    if dir:
        os.makedirs(dir, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(args.output_corr_file, sep='\t', index=False)

    print("Done!")