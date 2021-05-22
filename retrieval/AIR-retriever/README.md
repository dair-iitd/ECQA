# AIR-retriever
The AIR retriever code is taken from https://github.com/vikas95/AIR-retriever.git for Multi-Hop QA - ACL 2020 paper: [Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://arxiv.org/abs/2005.01218)

## Running Experiments:

1] Download the GLoVe embeddings 'glove.6B.100d.txt' in this folder.

2] Modify the 'Compute_IDF.py' file line 61. This file computes the Inverse Document Frequency weights to 'MultiRC_IDF_vals.json' file. Give the input_files as a list with the list element being the location of the file to be tested by AIR-retriever. For example: input_files = \["../air_test.json"\]

3] Run 
```bash
"python3 Compute_IDF.py"
```
4] Set the path to the input and output files in 'AIR_evidence_retrieval_scores.py' in lines 49 and 50. For example: input_file_name = '../air_test.json' and output_file_name = 'air_test_single_chain.tsv'. Then run 
```bash
"python3 AIR_evidence_retrieval_scores.py" to generate the output_file in the tsv format. 
```
3] Use the "retrieval_eval.py" script to generate the Recall, Precision and F1 scores for different metrics (STS-BERT, Spice, CIDEr and ROUGE). Before running this script, set up the spice and cider folders required for evaluation as specified in the generation folder's README.
```bash
"python3 retrievel_eval_AIR.py -o output_file -t ../data/E2_test.json"
For example: python3 retrievel_eval_AIR.py -o ./MultiRC_BM25_vs_POCC_justification_quality_score/air_test_single_chain.tsv -t ../data/E2_test.json
```
