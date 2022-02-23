import re
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import AffinityPropagation
from distance import jsd, div, cosine_distance, pdis, pdiv
from cluster import AffinityPropagationPosteriori as APP, IncrementalAffinityPropagation as IAPNA


def divergence_from_cluster_labels(labels1, labels2):
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))

    t1 = np.array([counts1[i] for i in n_senses])
    t2 = np.array([counts2[i] for i in n_senses])

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = t1/t1.sum()
    t2_dist = t2/t2.sum()

    return jsd(t1_dist, t2_dist)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='english', const='all', nargs='?',
                        help='Choose a language', choices=['english', 'latin'])
    parser.add_argument('--one_embedding_per_sentence', action='store_true',
                        help='If True, only keep embedding of the first target word in the sentence')
    parser.add_argument('--semeval_results', default='results_bert_english/', type=str,
                        help='Path to output results dir')
    parser.add_argument('--embeddings_path', default='embeddings_bert_english.pickle', type=str,
                        help='Path to output pickle file containing embeddings.')
    parser.add_argument('--trim', default=0, type=int,
                        help='Trim factor for Affinity Propagation a Posteriori')
    args = parser.parse_args()

    oneEmbPerSentence = args.one_embedding_per_sentence
    results_dir = args.semeval_results

    # language
    lang = args.language
    languages = ['english', 'latin']
    if lang not in languages:
        print('Language not valid, valid choices are: ', ', '.join(languages))
        sys.exit()

    bert_embeddings = pickle.load(open(args.embeddings_path, 'rb'))
    target_words = list(bert_embeddings.keys())
    results_dict = {'word': [], 'freq': [],
                    'ap_jsd': [], 'ap_pdis': [], 'ap_pdiv': [], 'ap-n_clusters':[],
                    'cd':[], 'div':[],
                    'iapna_jsd': [],  'iapna_pdis': [], 'iapna_pdiv': [], 'iapna-n_clusters':[],
                    'app_jsd': [], 'app_pdis': [], 'app_pdiv': [], 'app-n_clusters':[]}

    sentence_dict = dict()
    ap_dict, iapna_dict, app_dict = dict(), dict(), dict()

    print('Clustering BERT embeddings')
    for i, word in enumerate(target_words):
        print('\n=======', i+1, '- word:', word.upper(), '=======')
        emb = bert_embeddings[word]

        embeddings1 = []
        embeddings2 = []
        texts1 = []
        texts2 = []

        regex = r'\b%s\b' %word.replace('_vb', '').replace('_nn', '')

        time_slices = ['t1', 't2']
        for ts in time_slices:
            text_seen = {}

            for idx in range(len(emb[ts])):
                ts_text = ts + '_text'
                e = emb[ts][idx]
                text = emb[ts_text][idx]

                if not(re.search(regex, text)):
                    continue

                if oneEmbPerSentence:
                    if text in text_seen:
                        continue
                    else:
                        text_seen[text] = 1

                if ts == 't1':
                    embeddings1.append(e)
                    texts1.append(text)
                elif ts == 't2':
                    embeddings2.append(e)
                    texts2.append(text)


        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)

        freq = embeddings1.shape[0] + embeddings2.shape[0]
        print('t1 num. occurences: ', embeddings1.shape[0])
        print('t2 num. occurences: ', embeddings2.shape[0])
        print('tot num. occurrences:', freq)

        sentence_dict[word] = {time_slices[0]: texts1, time_slices[1]: texts2}

        # baselines: CD, DIV
        cd_dist = cosine_distance(np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0))
        print('CD:', cd_dist)
        div_dist = div(embeddings1, embeddings2)
        print('DIV:', div_dist)

        # Affinity Propagation (AP)
        embeddings_concat = np.concatenate([embeddings1, embeddings2], axis=0)
        ap = AffinityPropagation()
        ap.fit(embeddings_concat)
        labels = ap.labels_
        X1_labels, X2_labels = list(labels[:embeddings1.shape[0]]), list(labels[embeddings1.shape[0]:])
        ap_n = np.unique(labels).shape[0]
        print('AP - # clusters:', ap_n)
        ap_jsd = divergence_from_cluster_labels(X1_labels, X2_labels)
        print('AP - jsd:', ap_jsd)
        ap_pdis = pdis(embeddings1, embeddings2, X1_labels, X2_labels)
        print('AP - pdis:', ap_pdis)
        ap_pdiv = pdiv(embeddings1, embeddings2, X1_labels, X2_labels)
        print('AP - pdiv:', ap_pdiv)
        # add results to dataframe for saving
        ap_dict[word] = {time_slices[0]: X1_labels, time_slices[1]: X2_labels}

        # Incremental Affinity Propagation (IAPNA)
        iapna = IAPNA()
        iapna.fit(embeddings1)
        iapna.fit(embeddings2)
        labels = iapna.labels_
        X1_labels, X2_labels = list(labels[:embeddings1.shape[0]]), list(labels[embeddings1.shape[0]:])
        iapna_n = np.unique(labels).shape[0]
        print('IAPNA - # clusters:', iapna_n)
        iapna_jsd = divergence_from_cluster_labels(X1_labels, X2_labels)
        print('IAPNA - jsd:', iapna_jsd)
        iapna_pdis = pdis(embeddings1, embeddings2, X1_labels, X2_labels)
        print('IAPNA - pdis:', iapna_pdis)
        iapna_pdiv = pdiv(embeddings1, embeddings2, X1_labels, X2_labels)
        print('IAPNA - pdiv:', iapna_pdiv)
        # add results to dataframe for saving
        iapna_dict[word] = {time_slices[0]: X1_labels, time_slices[1]: X2_labels}


        # Affinity Propagation a Posteriori (APP)
        app = APP(trim=args.trim)
        app.fit(embeddings1)
        app.fit(embeddings2)
        labels = app.labels_
        embeddings1, embeddings2 = app._prev_X_trim, app._curr_X_trim
        X1_labels = list(labels[:embeddings1.shape[0]])
        X2_labels = list(labels[embeddings1.shape[0]:])
        app_n = np.unique(labels).shape[0]
        print('APP - # clusters:', app_n)
        app_jsd = divergence_from_cluster_labels(X1_labels, X2_labels)
        print('APP - jsd:', app_jsd)
        app_pdis = pdis(embeddings1, embeddings2, X1_labels, X2_labels)
        print('APP - pdis:', app_pdis)
        app_pdiv = pdiv(embeddings1, embeddings2, X1_labels, X2_labels)
        print('APP - pdiv:', app_pdiv)
        # add results to dataframe for saving
        app_dict[word] = {time_slices[0]: X1_labels,
                          time_slices[1]: X2_labels,
                          time_slices[0]+'_embeddings': embeddings1,
                          time_slices[1]+'_embeddings': embeddings2}


        results_dict['word'].append(word)
        results_dict['freq'].append(freq)
        results_dict['cd'].append(cd_dist)
        results_dict['div'].append(div_dist)
        results_dict['ap_jsd'].append(ap_jsd)
        results_dict['ap_pdiv'].append(ap_pdiv)
        results_dict['ap_pdis'].append(ap_pdis)
        results_dict['ap-n_clusters'].append(ap_n)
        results_dict['iapna_jsd'].append(iapna_jsd)
        results_dict['iapna_pdiv'].append(iapna_pdiv)
        results_dict['iapna_pdis'].append(iapna_pdis)
        results_dict['iapna-n_clusters'].append(iapna_n)
        results_dict['app_jsd'].append(app_jsd)
        results_dict['app_pdiv'].append(app_pdiv)
        results_dict['app_pdis'].append(app_pdis)
        results_dict['app-n_clusters'].append(app_n)

        os.makedirs(results_dir, exist_ok=True)

        # save results to CSV
        csv_file = results_dir + 'results_' + lang + '.csv'
        results_df = pd.DataFrame.from_dict(results_dict)
        results_df = results_df.sort_values(by=['word'], ascending=False)
        results_df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)

        # save cluster labels to pickle
        labels_file = results_dir + 'ap_labels_' + lang + '.pkl'
        pf = open(labels_file, 'wb')
        pickle.dump(ap_dict, pf)
        pf.close()

        # save cluster labels to pickle
        labels_file = results_dir + 'iapna_labels_' + lang + '.pkl'
        pf = open(labels_file, 'wb')
        pickle.dump(iapna_dict, pf)
        pf.close()

        # save cluster labels to pickle
        labels_file = results_dir + 'app_labels_' + lang + '.pkl'
        pf = open(labels_file, 'wb')
        pickle.dump(app_dict, pf)
        pf.close()

        # save sentences
        sents_file = results_dir + 'sents_' + lang + '.pkl'
        pf3 = open(sents_file, 'wb')
        pickle.dump(sentence_dict, pf3)
        pf3.close()
        print('Done! Saved results in', csv_file, '!')