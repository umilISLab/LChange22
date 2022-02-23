# LChange22

Official repository for LChange22 - WIDID. Published results were produced in Python 3 programming environment on Windows 11 operating system. Instructions for installation assume the usage of PyPI package manager.<br/>

Our code has been obtained starting from the SemEval submission (https://aclanthology.org/2020.semeval-1.6/) repository by Martinc, M. et al. (2020) available in GitHub (https://github.com/smontariol/Semeval2020-Task1). 

## Installation, documentation ##

Install dependencies if needed: pip install -r requirements.txt

### To reproduce the results published in the paper run the code in the command line using following commands: ###

#### Preprocess corpus:<br/>

Remove artifacts and numbers from SemEval English and Latin corpora.  In addition, we sanitize documents to make them less than 512 characters long. 
We use a recursive function to split a document about every 500 characters. No words is split in each document split.
 
```
python preprocessing.py  --corpus_paths pathToCorpusSlicesSeparatedBy';' --output_corpus_paths pathToOutputCorpusSlicesSeparatedBy';' --language language
```

#### Extract BERT embeddings:<br/>

Extract embeddings from the preprocessed corpora in .txt format:<br/>

```
python extract_embeddings.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --embeddings_path pathToOutputEmbeddingFile --concat 
```

This creates a pickled file containing all contextual embeddings for all target words.<br/>

#### Train and Extract Doc2Vec embeddings:<br/>

Train Doc2Vec and Extract embeddings from the preprocessed corpora in .txt format:<br/>

```
python doc2vec_extraction.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --embeddings_path pathToOutputEmbeddingFile --model_path pathToOutputModelFile 
```

This creates a pickled file containing all contextual embeddings for all target words.<br/>

#### Get results:<br/>

Conduct clustering and measure semantic shift with CD, DIV, JSD, PDIS, PDIV:<br/>

```
python calculate_semantic_change.py --language language --embeddings_path pathToInputEmbeddingFile --semeval_results pathToOutputResultsDir --trim trimValueForAPP
```

This script takes the pickled embedding file as an input and creates several files, a csv file containing CD, DIV, JSD, PDIS, PDIV scores for all clustering methods for each target word, files containing cluster labels for each embedding , and a file containing context (sentence) mapped to each embedding.<br/>

Generate CSV files for evaluation using Spearman's Correlation:<br/>

```
python evaluation.py --language language --results_file pathToInputResultsCSVFile --gold_path pathToSemEvalGoldFile --output_corr_file pathToOutputCorrelationFile
```

This script takes the CSV file generated in the previous step as an input and compute the Spearman's Correlation all the experimented methods ('cd', 'div', 'ap_jsd', 'ap_pdis', 'ap_pdiv', 'iapna_jsd', 'iapna_pdis', 'iapna_pdiv', 'app_jsd', 'app_pdis', 'app_pdiv').<br/>

#### If something is unclear, check the default arguments for each script. If you still can't make it work, feel free to contact us :).

