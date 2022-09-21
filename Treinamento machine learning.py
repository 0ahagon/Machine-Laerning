#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report,                            accuracy_score, confusion_matrix, auc

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# # About data set
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
# this date.The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no/less chance of heart attack and 1 = more chance of heart attack
# 
# Attribute Information
# 1) age
# 
# 2) sex (1 = male; 0 = female) 
# 
# 3) chest pain type (4 values)
# 
# 4) resting blood pressure
# 
# 5) serum cholestoral in mg/dl
# 
# 6)fasting blood sugar > 120 mg/dl
# 
# 7) resting electrocardiographic results (values 0,1,2)
# 
# 8) maximum heart rate achieved
# 
# 9) exercise induced angina
# 
# 10) oldpeak = ST depression induced by exercise relative to rest
# 
# 11)the slope of the peak exercise ST segment
# 
# 12) number of major vessels (0-3) colored by flourosopy
# 
# 13) thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
# 
# 14) target: 0= less chance of heart attack 1= more chance of heart attack

# In[2]:


dado = pd.read_csv('DADOS/heart.csv')
dado.sample(5)


# In[3]:


colunas_traducao = {
'age' : 'idade',
'sex' : 'sexo'  ,     
'cp' : 'tp_dor_tox'   ,
'trestbps' : 'pres_arterial'          ,
'chol' : 'colestoral_sérico'         ,      
'fbs' : 'glicemia_jejum',    
'restecg' : 'result_eletrocardio'   ,
'thalach' : 'frequ_card_max'            ,
'exang' : 'angina_induzida',
'oldpeak' : 'pico_antigo'  ,
'slope' : 'incli_seg_ST'      ,
'ca' : 'nu_vaso_principais',
'thal' : 'thal',
'target' :  'target'
}


# In[4]:


dado = dado.rename(columns = colunas_traducao)


# In[5]:


dado.sample(5)


# # QUAL SEXO TEM MAIOR CHANCE DE TER ?

# In[6]:


# Transforma sexo em categorico
dado['sexo'] = dado['sexo'].astype('category')


# In[ ]:





# In[7]:


modelo = smf.glm(formula='target ~ sexo + idade + thal + nu_vaso_principais + tp_dor_tox', data=dado,
                family = sm.families.Binomial()).fit()
print(modelo.summary())


# A função glm() ajusta-se aos modelos lineares generalizados, uma classe de modelos que inclui regressão logística. A sintaxe da função glm() é semelhante à de lm(), exceto que devemos passar o argumento family=sm.families.Binomial() para dizer ao python para executar uma regressão logística em vez de algum outro tipo de modelo linear generalizado.

# In[8]:


print(np.exp(modelo.params[1:])) # a probabilidade de ser/fazer sobre a de não ser/fazer- qq valor abaixo de 1 diminui a chance de ser/ fazer acima de 1 aumenta


# interpretando os dados : homens possuem 0.23 chance de terem ataque  do coração
# 
#     para cada ano a mais o individuo tem 0.94 chance que outro individuo com um ano a menos

# In[9]:


(np.exp(modelo.params[1:]) / (np.exp(modelo.params[1:]) +1 )) * 100  #probabilidade de fazer apenas


# Podemos também gerar os mesmos dados em percentuais relativos de chances para compará-los e obter uma interpretação parecida com a interpretação da regressão linear, mas em termos de chances.
# 
# 
# 
# Homens possuem 76 % menos chance do que mulheres de terem um ataque do coração
# Para cada ano a mais de idade, as chances diminuem 5.06%.

# # utilizando scikit-learn 

# In[ ]:





# In[10]:


X = dado[['idade','sexo','tp_dor_tox','pres_arterial','colestoral_sérico','glicemia_jejum',    
'result_eletrocardio'   ,
'frequ_card_max'            ,
'angina_induzida',
'pico_antigo'  ,
'incli_seg_ST'      ,
 'nu_vaso_principais',
 'thal']]
Y= dado.target# Criando conjunto de treino e teste


# In[11]:


model = LogisticRegression(penalty='none', solver='newton-cg')
model.fit(X,Y)


# In[12]:


dado.corr()


# In[13]:


importance = model.coef_[0]
#importance is a list so you can plot it. 
feat_importances = pd.Series(importance)
feat_importances.nlargest(20).plot(kind='barh',title = 'Feature Importance')


# In[14]:


dado


# * Um coeficiente de regressão descreve o tamanho e a direção da relação entre um preditor e a variável de resposta. Coeficientes são os números pelos quais os valores do termo são multiplicados em uma equação de regressão. 

# In[15]:


model = LogisticRegression(penalty='none', solver='newton-cg')
baseline_df = dado[['target', 'tp_dor_tox','result_eletrocardio','incli_seg_ST']].dropna()
Y = baseline_df.target
X = pd.get_dummies(baseline_df[['tp_dor_tox','result_eletrocardio','incli_seg_ST']], drop_first=True)
print(X)


# In[16]:


model.fit(X, Y)


# In[17]:


print(model.coef_) # Temos o mesmo modelo!


# In[18]:


# Predizendo as probabilidades
yhat = model.predict_proba(X)


# In[19]:


yhat = yhat[:, 1] # manter somente para a classe positiva


# In[20]:


confusion_matrix(Y, model.predict(X)) # usando a função do sklearn


# In[21]:


pd.crosstab(Y, model.predict(X))  # fazendo "na mão"


# In[22]:


acuracia = accuracy_score(Y, model.predict(X))
print('O modelo obteve %0.4f de acurácia.' % acuracia)


# testando com variaveis de correlação(importância) negativa

# In[23]:


model = LogisticRegression(penalty='none', solver='newton-cg')
baseline_df = dado[['target', 'sexo','angina_induzida','thal','nu_vaso_principais','pico_antigo']].dropna()
Y = baseline_df.target
X = pd.get_dummies(baseline_df[['sexo','angina_induzida','thal','nu_vaso_principais','pico_antigo']], drop_first=True)
print(X)


# In[ ]:





# In[24]:


model.fit(X, Y)


# In[25]:


# Predizendo as probabilidades
yhat = model.predict_proba(X)


# In[26]:


yhat = yhat[:, 1] # manter somente para a classe positiva


# In[27]:


confusion_matrix(Y, model.predict(X)) # usando a função do sklearn


# In[28]:


acuracia = accuracy_score(Y, model.predict(X))
print('O modelo obteve %0.4f de acurácia.' % acuracia)


# testando com as variaveis de maior importancia seja negativas ou positivas

# In[29]:


model = LogisticRegression(penalty='elasticnet', solver='saga',C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=0, max_iter=100,
                   multi_class='warn', n_jobs=None,
                   random_state=None,tol=0.0001, verbose=0,
                   warm_start=False)
baseline_df = dado[['target', 'tp_dor_tox','result_eletrocardio','incli_seg_ST','sexo','angina_induzida','thal','nu_vaso_principais','pico_antigo']].dropna()
Y = baseline_df.target
X = pd.get_dummies(baseline_df[['tp_dor_tox','result_eletrocardio','incli_seg_ST','sexo','angina_induzida','thal','nu_vaso_principais','pico_antigo']], drop_first=True)
print(X)


# In[30]:


model.fit(X, Y)


# In[31]:


acuracia = accuracy_score(Y, model.predict(X))
print('O modelo obteve %0.4f de acurácia.' % acuracia)


# A acurácia não é uma medida muito boa para trabalhar com modelos de classificação pois ela pode nos induzir a achar que um modelo que prediz corretamente todos os zeros mas nenhum um é um modelo muito bom. Vejamos outras alternativas de métricas mais interessantes do que a acurácia.

# In[32]:


print(classification_report(Y, model.predict(X)))


# O classification report do SciKit-Learn nos provê as três métricas de avaliação apresentadas na figura acima.
# 
# Precision é a capacidade do modelo de não prever uma instância negativa como positiva (não cometer erro do tipo 1). Para todas as instância classificadas como positivas, qual é o percentual de acerto.
# 
# Recall é a capacidade do modelo de encontrar todas as instâncias positivas. Para todas as instâncias que são de fato positivas, qual é o percentual de acerto.
# 
# A métrica F1 conjuga as duas anteriores como uma média harmônica entre ambas. Ela deve sempre ser priorizada para comparar modelos de classificação em relação à acurácia.
# 
# Uma excelente alternativa é fazer a curva ROC e calcular o AUC (área debaixo da curva).
# 
# A curva ROC (Receiver Operating Characteristic Curve) leva em conta a TPR (True Positive Rate ou Recall ou Sensitity) e a FPR (False Positive Rate ou Specificity).
# 
# A curva ROC traça esses dois parâmetros. o AUC (Area Under the Curve) é um valor que sintetiza a informação da curva ROC. Ela varia de 0.5 a 1. Em suma, essa métrica nos diz o quanto o modelo é capaz de distinguir as duas classes. Vejamos o AUC e a curva RUC para o modelo que estimamos.

# In[33]:


print('AUC: %0.2f' % roc_auc_score(Y, yhat))


# In[34]:


def plot_roc_curve(y_true, y_score, figsize=(10,6)):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=figsize)
    auc_value = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[35]:


plot_roc_curve(Y, yhat)


# Podemos considerar uma área debaixo da curva de mais de 0.7 como aceitável. Mais de 0.8 parece bom. Mais de .9 está excelente. Há também outras métricas que podemos explorar.

# # Predições

# In[36]:


dado.sample(5)


# In[37]:


eu = pd.DataFrame({'tp_dor_tox':2,'result_eletrocardio':0,'incli_seg_ST':1,'sexo':1,'angina_induzida':1,'thal':2,'nu_vaso_principais':0,'pico_antigo':3.0}, index=[0])
minha_prob = model.predict_proba(eu)
print('Eu teria {}% de ter um ataque'      .format(round(minha_prob[:,1][0]*100, 2)))


# tentar prever quem terá um ataque 

# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)# Treinando modelo


# In[39]:


model.fit(X_train, y_train)


# In[40]:


X_test[10:16]


# In[41]:


lista = []
for i in range(len(X_test)):
    minha_prob = model.predict_proba(X_test)
    prob = (minha_prob[i])
    print('Eu teria {}% de ter um ataque'      .format(round(prob[1:2][0]*100, 2)))


# In[42]:


dado[10:16]


# In[ ]:





# In[43]:


dado.query(' tp_dor_tox==2 and result_eletrocardio == 0 and incli_seg_ST == 2 and angina_induzida==0 and thal == 2 and nu_vaso_principais==0 and pico_antigo==0.0 and sexo==1')


# In[44]:


dado.query(' tp_dor_tox==2 and result_eletrocardio == 1 and incli_seg_ST == 2 and angina_induzida==0 and thal == 2 and nu_vaso_principais==0 and pico_antigo==1.6 and sexo==1')


# http://neylsoncrepalde.github.io/2019-11-25-regressao_logistica_python/

# In[45]:


def freq(x: pd.Series, plot=False):
    contagem = x.value_counts()
    percentual = round((x.value_counts() / x.shape[0]) * 100, 3)
    res = pd.DataFrame({'values': x.unique(), 'n': contagem, 'perc': percentual})
    if plot:
        sns.countplot(x)
        plt.show()
    return res


# In[46]:


# Quantos sobreviveram e não sobreviveram
(dado.target.value_counts() / dado.shape[0]) * 100


# In[47]:


freq(dado.sexo, plot=True)


# # escolhendo as variaveis para o modelo de Classificação

# podemos tentar identificar se há correlação entre as variavéis

# In[ ]:





# In[48]:


from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression


# In[ ]:





# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns# Carregando dataset

X2 = dado[['idade','sexo','tp_dor_tox','pres_arterial','colestoral_sérico','glicemia_jejum',    
'result_eletrocardio'   ,
'frequ_card_max'            ,
'angina_induzida',
'pico_antigo'  ,
'incli_seg_ST'      ,
 'nu_vaso_principais',
 'thal']]
Y2 = dado.target# Criando conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X2, Y2, random_state=42)# Treinando modelo
model_randomforest  = RandomForestClassifier()
model_randomforest.fit(X_train, y_train)# Mostrando importância de cada feature
model_randomforest.feature_importances_


# In[50]:


0.1119876 + 0.03353875+ 0.1259624 +0.05801383+ 0.09194341+0.01479495+ 0.01201865+ 0.0998143 + 0.0539836 + 0.11418677+0.05222034+ 0.17040528+0.06113012


#  Se  somar todos os elementos , verá que o resultado será 1 ou próximo de 1. Ao analisar esse array, podemos ver que a feature mais importante para o algoritmo foi a vigéssima segunda, nu_vaso_principais. Se quiser gerar uma visualização para ver as features mais importantes, o código é similar a esse:

# In[52]:


importances = pd.Series(data=model_randomforest.feature_importances_, index=X2.columns)
sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')


# Cuidado com esse método! As vezes, os valores mostrados pelo feature_importances_ pode ser enviesado dependendo dos parâmetros definidos na criação do objeto. O que isso quer dizer? Evite usar os parâmetros default do RandomForestClassifier()

# SELECIONANDO FEATURES COM O KBEST
# KBEST É UM TESTE ESTÁTISTICO UNIVARIADO
# 
# 
#  Basta informar que quer selecionar apenas as K maiores features do seu dataset com base em um teste estatístico

# In[53]:


from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
X3 = dado[['idade','sexo','tp_dor_tox','pres_arterial','colestoral_sérico','glicemia_jejum',    
'result_eletrocardio'   ,
'frequ_card_max'            ,
'angina_induzida',
'pico_antigo'  ,
'incli_seg_ST'      ,
 'nu_vaso_principais',
 'thal']]
Y3 = dado.target = SelectKBest(chi2, k=3).fit_transform(X, Y)


# In[54]:


print(Y3[:5]) # variável recem criada para os 3 melhores recursos


# In[55]:


print(X.head()) # comparar os valores acima com o da base X, podemos vê que as melhores variáveis são tp_dor_tox, nu_vaso_principais e pico_antigo 


# A parte Ruim é que escolher o número K ideal muitas vezes é uma tarefa difícil

# https://paulovasconcellos.com.br/como-selecionar-as-melhores-features-para-seu-modelo-de-machine-learning-2e9df83d062a

# In[56]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
X3 = dado[['idade','sexo','tp_dor_tox','pres_arterial','colestoral_sérico','glicemia_jejum',    
'result_eletrocardio'   ,
'frequ_card_max'            ,
'angina_induzida',
'pico_antigo'  ,
'incli_seg_ST'      ,
 'nu_vaso_principais',
 'thal']]
Y3 = dado.target# 
model = LinearSVC()
rfe = RFE(model, step=1).fit(X3, Y3)


# PREDIÇÃO DO RandomForestClassifier()

# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X2, Y2, random_state=42)# Treinando modelo
model.fit(X_train, y_train)


# In[58]:


pred = model_randomforest.predict(X_test)


# In[79]:


X_test


# In[59]:


lista = []
for i in range(len(X_test)):
    minha_prob = model_randomforest.predict(X_test)
    prob = (minha_prob[i])
    print('predição {}'      .format(prob))


# In[60]:


lista = []
for i in range(len(X_test)):
    minha_prob = model_randomforest.predict_proba(X_test)
    prob = (minha_prob[i])
    print('Eu teria {}% de ter um ataque'      .format(round(prob[1:2][0]*100, 2)))


# In[61]:


print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, pred)))


# **Criando modelo com as melhores features segundo kbest**

# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns# Carregando dataset

X4 = dado[['tp_dor_tox', 'nu_vaso_principais', 'pico_antigo']]
Y4 = dado.target# Criando conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X4, Y4, random_state=42)# Treinando modelo
model_randomforest  = RandomForestClassifier()
model_randomforest.fit(X_train, y_train)# Mostrando importância de cada feature


# In[63]:


pred2 = model_randomforest.predict(X_test)


# In[64]:


print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, pred2)))


# **Criando modelo com as melhores features segundo feature_importances_**

# In[65]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns# Carregando dataset

X4 = dado[['nu_vaso_principais', 'pico_antigo']]
Y4 = dado.target# Criando conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X4, Y4, random_state=42)# Treinando modelo
model_randomforest  = RandomForestClassifier()
model_randomforest.fit(X_train, y_train)# Mostrando importância de cada feature


# In[66]:


pred3 = model_randomforest.predict(X_test)


# In[67]:


print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, pred3)))


# In[ ]:





#  o **feature_importances_** não foi muito acertivo ao determinar as melhores features, porém tivemos uma melhora na acurracia com as variáveis do kbeast

# 

# In[69]:


X4 = dado[['idade','sexo','tp_dor_tox','pres_arterial','colestoral_sérico','glicemia_jejum',    
'result_eletrocardio'   ,
'frequ_card_max'            ,
'angina_induzida',
'pico_antigo'  ,
'incli_seg_ST'      ,
 'nu_vaso_principais',
 'thal']]
Y4= dado.target# Criando conjunto de treino e teste


# In[72]:


model = LogisticRegression(penalty='none', solver='newton-cg')
baseline_df = dado[['target', 'tp_dor_tox', 'nu_vaso_principais', 'pico_antigo']].dropna()
Y4 = baseline_df.target
X4 = pd.get_dummies(baseline_df[['tp_dor_tox', 'nu_vaso_principais', 'pico_antigo']], drop_first=True)
print(X4)


# In[73]:


model.fit(X4,Y4)


# In[74]:


acuracia = accuracy_score(Y4, model.predict(X4))
print('O modelo obteve %0.4f de acurácia.' % acuracia)


# In[ ]:




