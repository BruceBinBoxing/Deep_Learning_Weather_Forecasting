Global AI Challenger-Weather Forecasting
==============================
> Using deep learning produces more accurate weather forecasts based on the observation and prediction of meteorological elements.

Our rank-3 (CCIT007) score indicates end-to-end seq2seq Deep Learning model with small feature engineering can greatly improve weather forecasting accuracy (This is our first competition project. If it is helpful, a star would be a big stimuli for us. Many thanks to *Huaishao Luo's* big help.)
### License
MIT
### Project Framework
I feel [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/) is helpful to organize project codes. Hope you also like it.
### Requirements
I test it on **MacOS** and **Ubuntu**. It is based on **Python 3.6.** Required packages like keras, tensorflow etc. are iincluded in **requirements.txt**. Run bellow command to install them.
> pip install -r requirements.txt
     
### Customize model parameters

You can dive into the './src/models/parameter_config_class.py'. Due to the parameters of the deep model are too many. Here we do not play exhaustedly. We mainly play different parameters of 'self.layers' and use ensemble to combine shallow and deep seq2seq model.

Hope your like this project. Please let me know your ideas and questions.


Best regards.

*Bin*

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
