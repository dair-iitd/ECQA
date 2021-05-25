# AIR-retriever
The AIR retriever code is taken from https://github.com/vikas95/AIR-retriever.git for Multi-Hop QA - ACL 2020 paper: [Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://arxiv.org/abs/2005.01218)

## Running Experiments:

1] Download the GLoVe embeddings 'glove.6B.100d.txt' in this folder. You can download it from [here](https://www.kaggle.com/danielwillgeorge/glove6b100dtxt)

2] Download 'wordnet' from nltk.
```bash
import nltk
nltk.download('wordnet')
```

2] Run the 'Compute_IDF.py' which computes the Inverse Document Frequency weights to 'MultiRC_IDF_vals.json' file. Give the input as the location of the file to be tested by AIR-retriever in following manner:
```bash
python3 Compute_IDF.py -i input_file
For example: python3 Compute_IDF.py -i ../air_test.json
```

3] Run AIR_evidence_retrieval_scores.py file with the input file as the file to be tested by AIR-retriever and output file to be in tsv format. The output directory is ./MultiRC_BM25_vs_POCC_justification_quality_score/ 
```bash
python3 AIR_evidence_retrieval_scores.py -i input_file -o output_file
For example: python3 AIR_evidence_retrieval_scores.py -i ../air_test.json -o air_test_output.tsv
```
## Evaluation

### Gold corpus

4] Use the "retrieval_eval_AIR_gold.py" script to generate the Exact Recall, Precision and F1 scores.
```bash
python3 retrievel_eval_AIR_gold.py -o output_file -t ../data/E2_test.json
For example: python3 retrievel_eval_AIR_gold.py -o ./MultiRC_BM25_vs_POCC_justification_quality_score/air_test_output.tsv -t ../data/E2_test.json
```

### Silver corpus

4] Use the "retrieval_eval_AIR.py" script to generate the Recall, Precision and F1 scores for different metrics (STS-BERT, Spice, CIDEr and ROUGE). Before running this script, set up all the folders required for evaluation as specified in the generation folder's README.
```bash
python3 retrievel_eval_AIR.py -o output_file -t ../data/E2_test.json
For example: python3 retrievel_eval_AIR.py -o ./MultiRC_BM25_vs_POCC_justification_quality_score/air_omcs_test_output.tsv -t ../data/E2_test.json
```
