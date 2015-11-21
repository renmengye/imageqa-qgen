# A Question Generator

## Usage
```
python question_generator.py -parser_path {Stanford parser path} \
                             -sentence {Single sentence} \
                             -list {List of sentences file} \
                             -parsed_file {Parsed file} \
                             -output {Output file}
```

## Prerequisites
1. You need to download NLTK WordNet package.
```
python
import nltk
nltk.download()
d
wordnet
```

2. You need to download Stanford Parser
at http://nlp.stanford.edu/software/lex-parser.shtml#Download

3. Extract the zip into a folder and remember the path

4. You need to copy *lexparser_sentence.sh* into the Stanford Parser folder.
```
cp lexparser_sentence.sh stanford-parser/lexparser_sentence.sh
```

## Examples
### Run a single sentence
```
python question_generator.py -sentence "A man is riding a horse"
```

### Run a list of sentences
* Provide a file with each line in the file to be a sentence.
* Output is a pickle file, storing a list.
* Each element in the list is a tuple of five fields:
    1. Original sentence ID (0-based)
    2. Original sentence
    3. Generated question
    4. Answer to the generated question
    5. Type of the generated question

```
python question_generator.py -list sentences.txt -output questions.pkl
```

### Run a pre-parsed file
Run stanford parser to pre-compute the parse trees.

```
lexparser.sh sentences.txt > sentences_parsed.txt
```

```
python question_generator.py -parsed_file sentences_parsed.txt \
                             -output questions.pkl
```

## Reference
*Exploring Models and Data for Image Question Answering*. Mengye Ren, Ryan
Kiros, Richard Zemel. NIPS, 2015.
```
@inproceedings{ren2015imageqa,
  title={Exploring Models and Data for Image Question Answering},
  author={Mengye Ren and Ryan Kiros and Richard Zemel},
  booktitle={NIPS},
  year={2015}
}
```
