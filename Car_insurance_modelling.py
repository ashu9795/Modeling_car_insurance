# Import required modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
cars=pd.read_csv("car_insurance.csv")

cars.isna()           # it wl convert all dataset in true or false nd NULL values is reprenseted                                  by NAN
cars.isna().any()         # it is for checking the value that in ech column any nULL value is                                    present or not if prsent it will priint true else false


cars.isna().sum()               # it will count  NULL value by column wise

                                # best_feature_df.isna().sum().plot(kind="bar")

                                # best_feature_df.show() it will convert in graph for better                                            understanding
cars.info()                                     # it is for displaying not null column


credit_score_mean=cars["credit_score"].mean()   # it is for calculating mean of that column
annual_mileage_mean=cars["annual_mileage"].mean()

cars["credit_score"].fillna(credit_score_mean,inplace=True)   # fiing missing value for  with mean
cars["annual_mileage"].fillna(annual_mileage_mean,inplace=True)

models=[]
features=cars.drop(columns=["id","outcome"]).columns          # it is taking a the recods of                    the                                                                    column except the id ans outcome. 
for i in features :
    model=logit(f"outcome~{i}",data=cars).fit()             # it will create a lgistic regression                                                                 model with the element of fetures and                                                                 and the outcome and store in model.
    models.append(model)
accuracies=[]
for i in range(0,len(models)):
    conf=models[i].pred_table()  # it will calculate the accurance of a particular column by                                            confusion matrix
    tn=conf[0,0]
    tp=conf[1,1]
    fn=conf[1,0]
    fp=conf[0,1]
    acc=(tn+tp)/(tn+fn+fp+tp)
    accuracies.append(acc)
best_feature=features[accuracies.index(max(accuracies))]    # it will give the index of hghest                                                                      accurancy and from index you get that                                                                                   feature.
    
best_feature_df = pd.DataFrame({"best_feature": best_feature,
                                "best_accuracy": max(accuracies)},index=[0])

sns.regplot(x="best_feature",y="best_accuracy",data = best_feature_df)
plt.show()

best_feature_df