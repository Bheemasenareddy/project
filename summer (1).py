#!/usr/bin/env python
# coding: utf-8

# In[237]:


import pandas as pd
df=pd.read_csv("C:\ml\project1\summer-products-with-rating-and-performance_2020-08.csv")


# In[238]:


df.head()


# In[239]:


df.columns


# In[240]:


df1=df[df.rating!=3.7]
df1.shape


# In[241]:


df1['rating']=df1['rating'].apply(lambda x:1 if (x > 3.7) else 0)
df1.head()


# In[242]:


df2=df1.groupby('rating').size()
df2


# In[243]:


per=df2/sum(df2)*100
per


# In[244]:


import matplotlib.pyplot as plt
df2.plot(kind='bar',title='Label Distribution')
plt.xlabel('rating')
plt.ylabel('values')
plt.show


# In[245]:


df3=df.iloc[:,[1,7]]
df3=df[df.rating!=3.7]
df3['rating']=df3['rating'].apply(lambda x : 1 if (x > 3.7) else 0)
df3


# In[246]:


df3


# In[247]:


df3['data']=df3['merchant_title'].str.len()
a=df3['data']
a


# In[248]:


df4=df3[['merchant_title','rating']]
df4


# In[249]:


df4['merchant_title']=df4['merchant_title'].str.len()
df4


# In[250]:


df4[["merchant_title",'rating']].mean()


# In[253]:


df4.plot(kind='bar',color='b',title='positive review')
plt.xlabel('rating')
plt.ylabel('title_orig')
plt.show()


# In[254]:


df4['ratio']=df3['price']/df3['retail_price']


# In[255]:


df4['ratio']=df4['ratio'].apply(lambda x:0 if(x<0.5) else 1)
df4['que']=df4.apply(lambda x:1 if ((x['rating'] and x['ratio'])==1) else 0,axis=1)
df4=df4.groupby('que').size()
df4


# In[256]:


df4.plot(kind='bar',color=['r','g'],title='review')
plt.xlabel('rating')
plt.ylabel('pricedata')
plt.show()


# In[257]:


df[['merchant_title', 'merchant_name']].head()


# In[258]:


df['urgency_text'].str.len()


# In[259]:


df.dropna(axis=0,inplace=True)
df


# In[260]:


df.info()


# In[261]:


df.columns


# In[262]:


df1=df[df.rating!=3.7]
df1.shape


# In[263]:


df1['rating']=df1['rating'].apply(lambda x:1 if (x > 3.7) else 0)
df1.head()


# In[264]:


df2=df1.groupby('rating').size()
df2


# In[265]:


per=df2/sum(df2)*100
per


# In[266]:


import matplotlib.pyplot as plt
df2.plot(kind='bar',color=['g','r'],title='Label Distribution')
plt.xlabel('rating')
plt.ylabel('values')
plt.show


# In[267]:


df3=df.iloc[:,[1,7]]
df3=df[df.rating!=3.7]
df3['rating']=df3['rating'].apply(lambda x : 1 if (x > 3.7) else 0)
df3


# In[268]:


df3['data']=df3['merchant_title'].str.len()
a=df3['data']
a


# In[269]:


df4=df3[['merchant_title','rating']]
df4


# In[270]:


df4['merchant_title']=df4['merchant_title'].str.len()
df4


# In[271]:


df4


# In[272]:


df4[["merchant_title",'rating']].mean()


# In[273]:


df4.plot(kind='bar',color='b',title='positive review')
plt.xlabel('rating')
plt.ylabel('title_orig')
plt.show()


# In[36]:


df4['ratio']=df3['price']/df3['retail_price']


# In[37]:


df4['ratio']=df4['ratio'].apply(lambda x:0 if(x<1) else 1)
df4['que']=df4.apply(lambda x:1 if ((x['rating'] and x['ratio'])==1) else 0,axis=1)
df4=df4.groupby('que').size()
df4


# In[38]:


df4.plot(kind='bar',color=['r','g'],title='review')
plt.xlabel('rating')
plt.ylabel('pricedata')
plt.show()


# In[39]:


data=df.iloc[:,[2,7,30]]
df5=data[df.rating!=3.7]
df5['rating']=df5['rating'].apply(lambda x:1 if(x>3.7) else 0)
df5.rating.value_counts()


# In[40]:


df5.shape


# In[41]:


df5.head()


# In[42]:


y=df5.rating
x=df5.merchant_title
x.shape


# In[43]:


y.shape


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[46]:


x_train.shape,x_test.shape


# In[47]:


y_train.shape,y_test.shape


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()


# In[49]:


x_train_dtm=vect.fit_transform(x_train)
x_train_dtm
print(type(x_train_dtm))
print(x_train_dtm.shape)
print(x_train_dtm)


# In[50]:


x_test_dtm=vect.transform(x_test)
x_test_dtm


# In[51]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()


# In[52]:


nb.fit(x_train_dtm, y_train)


# In[53]:


print(type(x_train_dtm))
print(x_train_dtm.shape)
print(x_test_dtm)


# In[54]:


y_pred_class=nb.predict(x_test_dtm)
y_pred_class


# In[55]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
gb=metrics.confusion_matrix(y_test,y_pred_class)
print(gb)


# In[56]:


import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(gb)
pl.title('confusion matrix')
pl.colorbar()
pl.show()


# In[57]:


from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred_class)
acc


# In[58]:


error=1-acc
error


# In[59]:


recall=(gb[1,1]+0.0)/sum(gb[1,:])
recall


# In[60]:


pre=(gb[1,1]+0.0)/sum(gb[:,1])
pre


# In[61]:


F1=(2*pre*recall)/(pre+recall)
F1


# In[62]:


print('Accuaracy',metrics.accuracy_score(y_test,y_pred_class))
print('recall',metrics.recall_score(y_test,y_pred_class))
print('precision',metrics.precision_score(y_test,y_pred_class))
print('F1-score',metrics.f1_score(y_test,y_pred_class))


# In[63]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_class,target_names=['Negative','Positive']))


# In[64]:


x_test[(y_pred_class==1)&(y_test==0)].count


# In[65]:


y_pred_prob=nb.predict_proba(x_test_dtm)[:,1]
y_pred_prob


# In[66]:


roc_auc=metrics.roc_auc_score(y_test,y_pred_prob)
roc_auc


# In[108]:


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import random
false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,y_pred_prob)
roc_auc=auc(false_positive_rate,true_positive_rate)
print('ROC RATE',roc_auc)
plt.title('ROC')
plt.plot(false_positive_rate,true_positive_rate,'g',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim(0.1,0.4)
plt.ylim(0.1,0.4)
plt.xlabel('True Positive Rate')
plt.ylabel('False Positve Rate')
plt.show()


# In[109]:


from sklearn.metrics import log_loss
log_error=log_loss(y_test,y_pred_prob)
log_error


# In[110]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[112]:


logreg.fit(x_train_dtm,y_train)


# In[113]:


y1_pred_class=logreg.predict(x_test_dtm)
y1_pred_class


# In[114]:


cm=metrics.confusion_matrix(y_test,y1_pred_class)
cm


# In[115]:


import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(cm)
pl.title('confusion matrix')
pl.colorbar()
pl.show()


# In[116]:


y1_pred_prob=logreg.predict_proba(x_test_dtm)[:,1]
y1_pred_prob


# In[117]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y1_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)


# In[119]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_lg)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim(0.1,0.4)
plt.ylim(0.1,0.4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[120]:


from sklearn.metrics import log_loss
log_error=log_loss(y_test,y1_pred_prob)
log_error


# In[121]:


print('Accuracy', metrics.accuracy_score(y_test, y1_pred_class))
print('Recall', metrics.recall_score(y_test,y1_pred_class))
print('Precision', metrics.precision_score(y_test,y1_pred_class))
print('F1 Score', metrics.f1_score(y_test,y1_pred_class))


# In[122]:


print(classification_report(y_test,y1_pred_class,target_names=['Negative','Positive']))


# In[126]:


from sklearn.linear_model import LogisticRegression
logreg1 = LogisticRegression(penalty='l2',C=1)
logreg1.fit(x_train_dtm, y_train)


# In[127]:


y2_pred_class = logreg1.predict(x_test_dtm)


# In[128]:


cml1= metrics.confusion_matrix(y_test, y2_pred_class)
cml1


# In[129]:


import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(cml1)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()


# In[131]:


y2_pred_prob = logreg.predict_proba(x_test_dtm)[:,1]
y2_pred_prob


# In[132]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y2_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)


# In[134]:


from matplotlib import pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_lg)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim(0.1,0.4)
plt.ylim(0.1,0.4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[135]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y2_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)


# In[136]:


print('Accuracy', metrics.accuracy_score(y_test, y2_pred_class))
print('Recall', metrics.recall_score(y_test,y2_pred_class))
print('Precision', metrics.precision_score(y_test,y2_pred_class))
print('F1 Score', metrics.f1_score(y_test,y2_pred_class))


# In[137]:


from sklearn.metrics import log_loss
log_error=log_loss(y_test, y2_pred_prob)
log_error


# In[138]:


x_train_tokens = vect.get_feature_names()
len(x_train_tokens)


# In[140]:


print(x_train_tokens)


# In[141]:


nb.feature_count_


# In[142]:


nb.feature_count_.shape


# In[143]:


neg_token_count = nb.feature_count_[0, :]
neg_token_count


# In[144]:


pos_token_count = nb.feature_count_[1, :]
pos_token_count


# In[146]:


tokens = pd.DataFrame({'token':x_train_tokens, 'Negative':neg_token_count, 'Positive':pos_token_count}).set_index('token')
tokens.head()


# In[147]:


tokens.sample(20, random_state=5)


# In[148]:


nb.class_count_


# In[149]:


tokens['Negative'] = tokens.Negative+ 1
tokens['Positive'] = tokens.Positive + 1
tokens.sample(5, random_state=6)


# In[150]:


tokens['Negative']= tokens.Negative / nb.class_count_[0]
tokens['Positive'] = tokens.Positive / nb.class_count_[1]
tokens.sample(5, random_state=6)


# In[151]:


tokens['Positive_ratio'] = tokens.Positive / tokens.Negative
tokens.sample(5, random_state=6)


# In[153]:


top=tokens.sort_values('Positive_ratio', ascending=False)
print(type(top))
print(top.shape)
print(top.head(20))


# In[154]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(x_train_dtm, y_train)


# In[155]:


print(type(x_train_dtm))
print(x_train_dtm.shape)


# In[157]:


ys_pred_class = clf.predict(x_test_dtm)
print(ys_pred_class.shape)


# In[158]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
csvm=metrics.confusion_matrix(y_test, ys_pred_class)
csvm


# In[160]:


import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(csvm)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()


# In[161]:


print('Accuracy', metrics.accuracy_score(y_test, ys_pred_class))
print('Recall',metrics.recall_score(y_test,ys_pred_class))
print('Precision' ,metrics.precision_score(y_test,ys_pred_class))
print('F1-Score',metrics.f1_score(y_test,ys_pred_class))


# In[162]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ys_pred_class,target_names=['Negative','Positive']))


# In[163]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ys_pred_class)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_auc)


# In[166]:


plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[167]:


from sklearn.metrics import log_loss
log_error=log_loss(y_test, ys_pred_class)
log_error


# In[170]:


from sklearn import linear_model, datasets


# In[172]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
Xlg = vect.fit_transform(x)
Xlg


# In[184]:


from sklearn.model_selection import cross_val_score


# In[210]:


logreg = LogisticRegression(C=100)
scores =cross_val_score(logreg, Xlg, y,cv=3,scoring='accuracy')
print(scores)


# In[211]:


print(scores.mean())


# In[212]:


logreg = LogisticRegression(penalty='l2',C=0.05)
scores = cross_val_score(logreg, Xlg, y, cv=2, scoring='accuracy')
print(scores)


# In[213]:


logrg = LogisticRegression(penalty='l2',C=0.05)
L_range = list(range(1,5))
L_scores = []
for l in L_range:
    logreg = LogisticRegression(C=l)
    scores = cross_val_score(logrg, Xlg, y, cv=10, scoring='accuracy')
    L_scores.append(scores.mean())
print(L_scores)


# In[214]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(L_range, L_scores)
plt.xlabel('Value of Reverse of regularizer for C ')
plt.ylabel('Cross-Validated Accuracy')


# In[215]:


from sklearn.model_selection import GridSearchCV


# In[216]:


l_range = list(range(1,5))
print(l_range)


# In[217]:


param_grid=dict(C=l_range)
print(param_grid)


# In[220]:


grid = GridSearchCV(logrg, param_grid, cv=3, scoring='accuracy')


# In[221]:


grid.fit(Xlg, y)


# In[227]:


grid.cv_results_


# In[236]:


grid_mean_scores = [result for result in grid.cv_results_]
print(grid_mean_scores)


# In[ ]:




