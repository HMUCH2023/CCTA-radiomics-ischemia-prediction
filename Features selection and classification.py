import pandas as pd
import numpy as np
import pymrmr
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV


def bootstrap(prob,pred,label, B, index):
    prob_array = np.array(prob)
    pred_array = np.array(pred)
    n = len(prob_array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        prob_sample = prob_array[index_arr]
        pred_sample = pred_array[index_arr]
        label_sample = label[index_arr]
        fpr,tpr, _ = roc_curve(label_sample,prob_sample)
        confusion_sample = confusion_matrix(label_sample,pred_sample)
        TP_sample = confusion_sample[1, 1]
        TN_sample = confusion_sample[0, 0]
        FP_sample = confusion_sample[0, 1]
        FN_sample = confusion_sample[1, 0]

        if index == 'auc':
            sample_result = metrics.auc(fpr,tpr)
        if index == 'acc':
            sample_result = (TP_sample+TN_sample)/(TP_sample+TN_sample+FP_sample+FN_sample)
        if index == 'sen':
            sample_result = TP_sample / float(TP_sample+FN_sample)
        if index == 'spe':
            sample_result = TN_sample / float(TN_sample+FP_sample)
        
        sample_result_arr.append(sample_result)
        
    sample_result_arr=np.array(sample_result_arr)
    
    mean=np.mean(sample_result_arr)
    std=np.std(sample_result_arr)

    lower = mean-1.96*std
    higher = mean+1.96*std
    return lower, higher

def model_classification(train_set,test_set):  
    x_train = train_set.iloc[:,1:]
    x_test = test_set.iloc[:,1:]
    y_train = train_set.iloc[:,0]
    y_test = test_set.iloc[:,0]
    
    #mRMR
    res = pymrmr.mRMR(x_train, 'MIQ', int(len(x_train.columns)*0.75))
    x_train_mrmr = x_train[res]
    
    #Lasso
    alphas = np.logspace(-2,3,50)
    model_lassoCV = LassoCV(alphas = alphas,cv = 10,max_iter = 100000,random_state = 42).fit(x_train_mrmr,y_train)
    coef = pd.Series(model_lassoCV.coef_, index = res)
    index_lasso = coef[coef != 0].index
    x_train_lasso = x_train[index_lasso]
    
    #RFE
    model = RFE(LogisticRegression(solver='liblinear',random_state = 42),step=1)
    param_grid = {'estimator__penalty': ['l1','l2'],'estimator__C': [1e-2, 1e-1, 1, 10],'n_features_to_select' :[4,5,6,7,8,9,10]}    
    grid_search = GridSearchCV(model, param_grid,cv=5)
    grid_search.fit(x_train_lasso, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()
    classifier = LogisticRegression(penalty = best_parameters['estimator__penalty'],C = best_parameters['estimator__C'],solver='liblinear',random_state = 42)
    selector = RFE(classifier, n_features_to_select = best_parameters['n_features_to_select'], step=1).fit(x_train_lasso, y_train)
   
        
    index_rfe = np.where(selector.ranking_ == 1)[0]
    x_train_rfe = x_train.iloc[:,index_rfe]
    x_test_rfe = x_test.iloc[:,index_rfe]
    classifier_model = classifier.fit(x_train_rfe, y_train)
    prob_train = classifier_model.predict_proba(x_train_rfe)
    prob_test = classifier_model.predict_proba(x_test_rfe)
    y_pred_train = classifier_model.predict(x_train_rfe)
    y_pred_test = classifier_model.predict(x_test_rfe)

    return prob_train,prob_test,y_pred_train,y_pred_test,y_test,y_train

B = int(1e6)
train_set = pd.read_excel(r"C:\Users\71915\Desktop\CQF.xlsx")   #the pathway of train set,label in the first column
test_set = pd.read_excel(r"C:\Users\71915\Desktop\CQF.xlsx")     #r'C:\Users\*' the pathway of test set,label in the first column


prob_train,prob_test,y_pred_train,y_pred_test,y_test,y_train = model_classification(train_set,test_set)


fpr, tpr, threshold = roc_curve(y_test, prob_test[:,1], pos_label = 1)
auc_score = roc_auc_score(y_test, prob_test[:,1])
confusion = confusion_matrix(y_test,y_pred_test)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
acc = (TP+TN)/(TP+TN+FP+FN)
sen = TP / float(TP+FN)
spe = TN / float(TN+FP)


index_list = ['auc','acc','sen','spe']
for index in index_list:
    if index == 'auc':
        print('auc:',auc_score)
    if index == 'acc':
        print('acc:',acc)
    if index == 'sen':
        print('sen:',sen)
    if index == 'spe':
        print('spe:',spe)
    print('CI:',bootstrap(prob_test[:,1],y_pred_test,y_test,B,index))