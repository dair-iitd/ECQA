# Retrieval Models

This section of the repository contains the instructions to preprocess the data in order to be fed into our deep retrieval pipeline which has been developed using [sbert](https://www.sbert.net/) library. We have given the code to train and run retrieval using the same.

## Data Preprocessing

First download the [OMCS Corpus file ](https://s3.amazonaws.com/conceptnet/downloads/2018/omcs-sentences-more.txt), which we refer to as the silver corpus, and paste  it in the data directory of the current root.

Obtain the final explanations data file, glued up with CQA data as mentioned in the [Data Section of the repositry](https://github.com/ShouryaAggarwal/Explanations-for-CommonSenseQA/tree/master/data), and paste in the data directory of the current root.

```bash
cp ../data/cqa_data.csv ./data/
```

Then run the following commands to generate the processed data for training and running inference on the models.

```bash
cd data
python3 ED_omcs_data_gen.py
python3 E2_data_generator.py
```


## BM-25

We use the python library of [rank-bm25](https://pypi.org/project/rank-bm25/) for this baseline. The following command will generate the naive top-k retrieved files, and the input files for AIR method using BM25 method.

### BM-25 Gold Corpus

```bash
python3 bm25_dump.py
```

### BM-25 Silver Corpus

```bash
python3 bm25_dump.py -test_omcs
```

## Deep SBERT based ranker

The following commands describe the training and retrieval procedure with a Sentence BERT based model, for the gold and silver corpus.

### Training

```bash
python3 IR_sbert_multi_dump.py -embedding_size 512 -model_save_dir <directory to save models>
```
You may change the embedding size as per your needs, the above parameter value was the one used to produce results in our ACL 2021 paper.
The above command would save the model to ```<directory to save models>/SBERT/multi_lr_2e-05_emb_512```, where ```2e-05``` is the learning reate (default) and ```512``` is the mentioned embedding size.

The below commands will generate the naive top-k retrieved files, and the input files for AIR method using a model trained by the above method.

### Retrieval with Gold Corpups

```bash
python3 IR_sbert_multi_dump.py -embedding_size 512 -test -pretrained_model <path to the pretrained model as explained above>
```

### Retrieval with Silver Corpups

```bash
python3 IR_sbert_multi_dump.py -embedding_size 512 -test -test_omcs -pretrained_model <path to the pretrained model as explained above>
```

## Evaluation
Do the setup as specified in the generation README for evaluation before running the following command. This would evaluate the input_file for top-k approach where k = 3 for positive properties and k = 1 for negative properties.
```bash
python retrieval_eval.py -i input_file
Example: python retrieval_eval.py -i 3_bm25_sets.json
```

## License
[Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
