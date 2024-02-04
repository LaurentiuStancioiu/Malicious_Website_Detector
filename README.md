## Malicious Website Detector

Training Dataset can be found [here](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)  
**Note**: We only trained on 10% of the data and obtained better results then all kaggle notebooks.   

## How we did it?

We got the embeddings of each website using the [ada embeddings](https://openai.com/blog/new-and-improved-embedding-model) from OpenAI and then tried multiple ML algorithms, sticked with the Random Forest Classifier.

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| benign     | 0.85      | 0.99   | 0.91     | 3018    |
| defacement | 0.91      | 0.90   | 0.91     | 2047    |
| malware    | 1.00      | 0.68   | 0.81     | 266     |
| phishing   | 0.99      | 0.44   | 0.61     | 669     |
| **accuracy**  |           |        | 0.88     | 6000    |
| **macro avg** | 0.94      | 0.75   | 0.81     | 6000    |
| **weighted avg** | 0.89      | 0.88   | 0.87     | 6000    |

As we can see above, the results do not look so well for malware and phishing. The application can be tried on [huggingface](https://huggingface.co/spaces/LaurentiuStancioiu/Malicious_website_detector). 

**!Attention!** Prone to error.

License for the dataset: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) 
