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

## Evaluation Setup

#### STS-BERT
We use the [semantic-text-similarity project](https://pypi.org/project/semantic-text-similarity/) to compute the STS-BERT scores. It is an easy-to-use interface to fine-tuned BERT models for computing semantic similarity between 2 sentences.

Install this project using:
```bash
pip install semantic-text-similarity
```
The web-based STS Bert model is downloaded by the 'generation_eval.py' script. Use 'cpu' or 'gpu' according to your machine specs in this script.

#### SPICE
You will first need to download the [Stanford CoreNLP 3.6.0](https://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run:
```bash
./get_stanford_models.sh
```
Note: SPICE will try to create a cache of parsed sentences in ./spice/cache/. This dramatically speeds up repeated evaluations. The cache directory can be moved by setting 'CACHE_DIR' in ./spice. In the same file, caching can be turned off by removing the '-cache' argument to 'spice_cmd'.

#### CIDEr
CIDEr evaluation code is taken from [Consensus-based Image Description Evaluation (CIDEr Code)](https://github.com/vrama91/cider).
First download their github repo:
```bash
git clone https://github.com/vrama91/cider
```
If you get unicode error, then go to the 'pyciderevalcap/tokenizer/ptbtokenizer.py' file, in the tokenize function of PTBtokenizer class, update the "prepare data for PTB Tokenizer" block  with this:
```bash
if self.source == 'gts':
  image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
  sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])
  sentences = sentences.encode('ascii', 'ignore').decode('ascii')
  final_tokenized_captions_for_image = {}

elif self.source == 'res':
  index = [i for i, v in enumerate(captions_for_image)]
  image_id = [v["image_id"] for v in captions_for_image]
  sentences = '\n'.join(v["caption"].replace('\n', ' ') for v in captions_for_image )
  sentences = sentences.encode('ascii', 'ignore').decode('ascii')
  final_tokenized_captions_for_index = []
```
#### METEOR
Follow meteor Readme for downloading one data file before evaluation. Use interactive notebook for calculating METEOR Scores.

## Evaluating Model Output
Do the setup as specified in the generation README for evaluation before running the following command. This would evaluate the input_file for top-k approach where k = 3 for positive properties and k = 1 for negative properties.
```bash
python retrieval_eval.py -i input_file
Example: python retrieval_eval.py -i 3_bm25_sets.json
```

## License
[Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
