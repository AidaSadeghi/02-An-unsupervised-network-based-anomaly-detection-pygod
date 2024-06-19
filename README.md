# 02-An-unsupervised-network-based-anomaly-detection-pygod


*1- Problem Understanding and Dataset Analysis:*

Understand the problem of financial fraud detection and its challenges.
Analyze the characteristics and structure of your dataset, including the features and the graph structure, if available.

*2- Data Preprocessing:*

A- Handle missing values, if any, through imputation or deletion.

B- Perform feature engineering and extraction to enhance the data representation.

WE DON'T NEED TO DO ANY rare class oversampling because we are doing the job unsupervised!
So we don't have any y in the first place to resample it!
 
 Balance the imbalanced dataset using techniques such as oversampling (e.g., SMOTE) or undersampling by trying different sampleres such as https://github.com/deepfindr/gnn-project/blob/697c77075d61040077f105db96e7443c8b34de59/oversample_data.py.
or samplers in pyG  https://pytorch-geometric.readthedocs.io/en/latest/modules/sampler.html#torch_geometric.sampler.BaseSampler
D- ADD NETWORK FEATURES for nodes and edges(Degree, first second order neighbor hood, adding some simple statistics of nodes): 
addin' attributes for edge and nodes (this can be done after)


NeighborSampler
ImbalancedSampler

PyGOD, outlier generator
https://github.com/pygod-team/pygod/tree/4d9b473de13b45d41b44e66f2d5c712c1a2aa72b/pygod/generator

*3- Graph Construction:*
Main construction is based on: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
https://www.youtube.com/watch?v=QLIkOtKS4os&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=12

Preprocessing with Pytorch Geometric Below, we show a few methods to loading data into pytorch geometric for training:
https://github.com/keitabroadwater/gnns_in_action/blob/master/chapter_2/GNNsInAction_Chapter_2.ipynb
Case C.
diffrent ways:
Create directly from a NetworkX graph object
Using raw files.
Using Dataset classes with raw files
Using Data objects to directly create a dataloader without the Dataset class

*4- Model Selection:*

after creating the object, we will use the https://github.com/pygod-team/pygod/blob/main/docs/examples/intro.py models to test the data
https://www.andrew.cmu.edu/user/yuezhao2/papers/22-neurips-adbench.pdf

Split your dataset into training, validation, and test sets.
Train your GNN model using the training data. Optimize the model's hyperparameters using techniques like grid search or Bayesian optimization.
Monitor the model's performance on the validation set and adjust the training process if necessary.

*5- Model Evaluation:*

Evaluate the performance of your trained GNN model on the test set using appropriate metrics for imbalanced datasets, such as precision, recall, F1-score, and area under the ROC curve (AUC-ROC).
Analyze the model's performance and interpret the results.

*6- Model Enhancement:*

Explore techniques to enhance further the model's performance, such as incorporating additional features or utilizing other machine learning algorithms in combination with the GNN.
Deployment and Monitoring with Optuna https://github.com/optuna/optuna
explainability features after the results

1- SHAP 
2- LIME 
3- GNN Explainer https://colab.research.google.com/drive/1s-mHf1_pKqohXj1pny-x-dqisCzMFB6A?usp=sharing

Checklist:
node attributes for ccf 
edge attributes for ccf
imbalanced data pre-processing 
same category for different. Datasets in writing this 
carbon-tracker 
Docker 
optuna 
lime 
shap 
gnnexplainer 
mlflow
oversampling minorclass 

*rerun the model 3 times*

1- with only original node features 

2- with both original features and network features

3- only network features- done - current run 

---------------------------------------------------------

to use the graph object in a supervised temporal study 

1- figure out the temporal dimension 

2- run built-in function to create the temporal object in PyG: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/temporal.html

-------------------------------------------------------
