# Fake_News_Detector_NLP_DeepLearning_Project
NLP Deep Learning Project that aims to detect fake news from the web


## Environment
To create your virtual environment, just install [conda](https://docs.anaconda.com/miniconda/miniconda-install/) and then run this in your terminal:

```shell
	conda env create --name=fake_news --file=environment.txt
```

You can also work with pip
```shell
	pip install requirements.txt
```

## Notebooks
There are 2 notebooks in this project.
One which finetunes the Fake News Detector model using Stella ("dunzhang/stella_en_1.5B_v5").
The second one is used to write a sentence in input and get a Fake or Real output.

## WARNING ##
 ° End of october 2024: T4 Tesla GPU from one day to the order couldn't run anymore flash-attn package. Make sure to use another GPU until it's corrected.
 
 ° For the interface it might occur an error if there is not enough time to load the model. In that case please wait a minute or two before trying again. 