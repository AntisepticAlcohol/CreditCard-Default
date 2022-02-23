#!/usr/bin/env python
# coding: utf-8

# In[9]:


## import modules
from flask import Flask
from flask import request, render_template

from keras.models import load_model
import joblib


# In[10]:


app = Flask(__name__) 


# In[11]:


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        Income = float(request.form.get("Income"))
        Age = float(request.form.get("Age"))
        Loan = float(request.form.get("Loan"))
        print(Income, Age, Loan)
        LR_model = joblib.load("LogisticReg")
        LR_pred = LR_model.predict([[Income, Age, Loan]])
        Tree_model = joblib.load("DecisionTree")
        Tree_pred = Tree_model.predict([[Income, Age, Loan]])
        RandomForest_model = joblib.load("RandomForest")
        RandomForest_pred = RandomForest_model.predict([[Income, Age, Loan]])
        XGBoost_model = joblib.load("XGBoost")
        XGBoost_pred = XGBoost_model.predict([[Income, Age, Loan]])
        NN_model = load_model("NeuralNet")
        NN_pred = NN_model.predict([[Income, Age, Loan]])
        print(LR_pred)
        print(Tree_pred)
        print(RandomForest_pred)
        print(XGBoost_pred)
        print(NN_pred)
        
        message1 = "Under logistic regression, the predicted default score is " + str(LR_pred)
        message2 = "Under decision tree, the predicted default score is " + str(Tree_pred)
        message3 = "Under random forest, the predicted default score is " + str(RandomForest_pred)
        message4 = "Under XGBoost, the predicted default score is " + str(XGBoost_pred)
        message5 = "Under neural network, the predicted default score is " + str(NN_pred)
        
        return(render_template("index.html", 
                               result1 = message1, 
                               result2 = message2, 
                               result3 = message3, 
                               result4 = message4, 
                               result5 = message5))
    else:
        return(render_template("index.html", 
                               result1 = "Hello! Please key some inputs to start predicting!", 
                               result2 = "", 
                               result3 = "", 
                               result4 = "", 
                               result5 = ""))


# In[12]:


if __name__ == "__main__":
    app.run()


# In[ ]:




