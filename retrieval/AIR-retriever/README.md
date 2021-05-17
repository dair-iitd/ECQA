# AIR-retriever
The AIR retriever code is taken from https://github.com/vikas95/AIR-retriever.git for Multi-Hop QA - ACL 2020 paper: [Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://arxiv.org/abs/2005.01218)

## Running Experiments:

1] Download the GLoVe embeddings 'glove.6B.100d.txt' in this folder.

2] Running "python3 AIR_evidence_retrieval_scores.py" shows the justification selection performance of AIR and will generate the output_file in the tsv format.

3] Use the "retrieval_eval.py" script to generate the Recall, Precision and F1 scores for different metrics (STS-BERT, Spice, CIDEr, METEOR and ROUGE). Before running this script, set up the spice and cider folders required for evaluation as specified in the generation folder's README.

"python3 retrievel_eval.py output_file ./E2_test.json"
