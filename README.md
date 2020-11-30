# Sentiment Analysis API using Avg-W2V and Deep Learning

**Structure:**

	- - sentimental Analysis
		- - SA_WebApp
			-- SA_models
				-- __init__.py
				-- TrainSA.ipynb
				-- airline_sentiment_analysis.csv
				-- text_process.py
				-- train.py
			-- __init__.py
			-- home.pt
			-- image_process.py
			-- views.py
		-- development.ini
		-- setup.py

1. First, run (.../sentimental Analysis/pip install -e .) this to install all the required libraries and files

2. Second, run (.../sentimental Analysis/pserve development.ini --reload) to start the server. However, if you haven't train the model then before running the second line goto (.../sentimental Analysis/SA_WebApp/SA_models) run python **train.py** to train your model by using the (**airline_sentiment_analysis.csv**). After training your model then run the second step to host the Sentimental Analysis API.

3. Then Open http://localhost:6543/ to see the web API.


#### Note:
* The file **.../sentimental Analysis/SA_WebApp/SA_models/TrainSA.ipynb**  will help you see which algorithm is better for the model.
* I tried different algorithms like LogisticRegression, SVM, Random Forest, Gaussian Naive Bayes, and two different Dense Neural Network (DNN) with Adam() and RMSprop() optimizer.
* As of now, I got the highest probability on Adam optimizer and I set Adam() as my main model optimizer in  train.py file.
* Max accuracy is between 84 - 86% until we set some random seed.
* Dont forget to incluse empty file *__init__.py* in **.../sentimental Analysis/SA_WebApp/SA_models/**

