#!/usr/bin/env python
# coding: utf-8

# # Partie pandas
Bonjour, bienvenue à notre devoir de groupe créer par Patrice-Emile et Veasna Aller hop ça va chauffer ! https://tenor.com/view/hello-there-private-from-penguins-of-madagascar-hi-wave-hey-there-gif-16043627La on affiche notre csv en tableau !
# In[241]:


import pandas as pd
import numpy as np
def csvReaderPlusPlus():
    return pd.read_csv("Dataset_7_price.csv", delimiter="|")

rd = csvReaderPlusPlus()
rd

on supprime les 10 premier colonnes inutiles, car les valeurs sont incorrectes ce qui permet d'éclaicir les données
# In[242]:


def dropColumnUseless(df):
    return df.drop(df.iloc[:,0:10],1,inplace=True)
    
dropColumnUseless(rd)
rd

on veut savoir combien de valeur null sont présents dans chaque colonne :) 
# In[243]:


def sumNull(rd):    
    return rd.isnull().sum()
sumNull(rd)

on additionne les valeurs pour chaque colonne différent puis on fait la moyenne, 
puis on remplace les valeurs null dans leurs colonnes spécifique bien sur(carat, deth, table)

Pour la qualité, on les a additionner ensemble chaque catégorie de qualité (Fair, Good, Very Good, Premium, Ideal) puis diviser par 2 pour obtenir le juste Milieu, puis on additionne les valeurs du plus mauvais qualité au plus bon pour tomber sur le chiffre que nous a donner le juste milieu, ce qui a permis de déterminer la moyenne, donc on est tomber sur premium alors on a remplacer les valeurs null par premium

(si ce n'est pas bon on c'est trompés on aurait du le diviser en 5 car il y a 5 qualités, on aurait mis à la place very good)
on a pas réussir a traduire ce moyen en code, donc on a courtounner donc on a remplacer les valeurs null par la moyenne qui est very good

pour les caractère spéciaux on a utiliser l'antislash "\" ;) pour les retirer on a galérer mine de rien :sob:
# In[244]:


def mettreMoyInNanCase(df,tabColumn,tabStr):
    
    for column in tabColumn :
        numeric = True
        if column in tabStr:
            numeric = False
        if numeric :
            somme = 0.0
            tableau = pd.to_numeric(df.loc[:,column], errors='coerce')
            tableau = tableau.replace(np.nan, 0, regex=True)
            for i in tableau :
                somme += i

            moy = somme/1400
            #print(column,"\t:\t",moy)

            df.loc[:,column] = df.loc[:,column].replace(np.nan, moy, regex=True)
        else:
            df.loc[:,column] = df.loc[:,column].replace(np.nan, tabStr[column], regex=True)
            df.loc[:,column] = df.loc[:,column].replace(['Allez au boulot ! :\)','None','\ù\*ùfsfsf///'], tabStr[column], regex=True)

#             tab = df[column].unique()
#             obj = {}
#             cpt = 0
#             for valTab in tab:
#                 obj[valTab] = 0
                
#             for valueColumn in df.loc[:,column]:
#                 for valueTab in tab:
#                     if valueTab == valueColumn:
#                         obj[valueTab] +=1
#                         cpt+=1 
            # print(obj)
            # print("cpt\t:\t",cpt,"\ncpt/len(tab)\t:\t",(cpt/len(tab)),"\ncpt/2\t:\t",(cpt/2))
            # moy = cpt/len(tab)
            
    return df.isnull().sum()
    
mettreMoyInNanCase(rd,['carat','cut','depth','table'],{'cut':'Very Good'})

tadaaaaa! on a réussir a supprimer les valeurs qui servent à rien héhé !
# In[245]:


print(rd['cut'].unique())

Pour nous ce qui est important est la qualité des carats, donc on va ranger la plus mauvais qualité à la plus bonne, 
car pour nous un client cherchera, la meilleure qualite ou la moyenne, contrairement au prix indique car avoir une bonne qualité 
dans tous les cas engendra un coût elevé.
Malheuresement on a pas assez de moyens pour se les payés 
# In[246]:


def trierParCut(df):
    df1 = df.loc[df.cut == "Fair"]
    df2 = df.loc[df.cut == "Good"]
    df3 = df.loc[df.cut == "Very Good"]
    df4 = df.loc[df.cut == "Premium"]
    df5 = df.loc[df.cut == "Ideal"]

    dfinal = pd.concat([df1,df2,df3,df4,df5])
    return dfinal

rd = trierParCut(rd)
display(rd)
print(rd['cut'].unique())

on regarde les types des colonnes présents
# In[247]:


def afficher(df):
    print(df.dtypes)
afficher(rd)

On affiches les statistiques de toutes les colonnes y compris float, int et object, c'est pour tout montrer
# In[248]:


def describeAll(df):
    d1 = rd.describe(include='all')
    print("les statistiques : \n")
    print(rd1)
describeAll(rd)

On affiches les statistiques des colonnes qui sont en float et int, on exclu les objects car ils ne servent à rien rip !
# In[249]:


def describeSpecific(df):
    rd1 = rd.describe(exclude=[object])
    print("les statistiques: \n")
    print(rd1)
describeSpecific(rd)

On modifie les colonnes en valeur numériques, mais on arrive pas à trouver une solution car la valeur 3 apparaît pas ! ou bien la valeur 4 disparaît et apparait le 3 comme par magie :( 
https://tenor.com/view/magic-shia-labeouf-snl-skit-fingers-gif-4860090
# In[250]:


def replaceCutValue(df,column):
    tab = df[column].unique()
    print(tab)
    for i in range(len(tab)):
        df.loc[:,column] = df.loc[:,column].replace(tab[i], (i+1), regex=True)

    print(df[column].unique())

replaceCutValue(rd, 'cut')

comme montre ci-dessous
# In[251]:


rd.loc[rd.cut == 3,:]

# ???

théoriquement ca fonctionne mais on maitrise pas pandas :sad:
# In[252]:


def AddNColonnes(df,column):
    
    tab = ['Fair','Good','Very Good','Premium','Ideal']
    
    for e in range(len(tab)):
        df.insert(len(df.columns), tab[e],0, allow_duplicates=False)

    for i in range(len(df.loc[:,column])):
        for v in range(len(df[column].unique())):
            if df.loc[:,column][i] == (v+1):
                df.loc[:,tab[v]][i] = 1


    return df.loc[:,(column,'Fair','Good','Very Good','Premium','Ideal')]

display(AddNColonnes(rd,'cut'))


# # partie Pyspark
Après quelque galères sur Pandas, on attaque PySpark ! https://tenor.com/view/just-do-it-shia-la-beouf-do-it-gif-4531935
# In[254]:


import pandas as pd
import pyspark
import numpy as np

file = 'Dataset_7_price.csv'

df = spark.read.options(delimiter='|').csv(file, header=True,inferSchema = True)

df = df.drop("_c0","Unnamed: 0","Unnamed: 0.1","Unnamed: 0.1.1","Unnamed: 0.1.1.1","Unnamed: 0.1.1.1.1","Unnamed: 0.1.1.1.1.1",
             "Unnamed: 0.1.1.1.1.1.1","Unnamed: 0.1.1.1.1.1.1.1","Unnamed: 0.1.1.1.1.1.1.1.1")

#df.select(df.columns[0:10],1,inplace=True)

def get_spark_env(version, job_name, n):
    if version in ["2.4"]:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[*]").config('spark.executor.memoryOverhead.config', n).appName(job_name).getOrCreate()
        return spark
    elif version in ["1.6"]:
        return None, None

def Print_DF(DF):
    return pd.DataFrame(DF.show(),columns=DF.columns)

def Print_DF_Rows(DF,NbRows):
    return pd.DataFrame(DF.take(NbRows),columns=DF.columns)

job_name = 'Formation'
spark = get_spark_env("2.4", job_name, '4g')
print (spark.version)

display(Print_DF_Rows(df, 10))

df.count()

Coup dur pour les joueurs français !! https://tenor.com/view/coup-dur-gif-13266651

# In[255]:


df.select(df.columns[6]).orderBy(["price"])


# In[256]:


df.orderBy(["color"], ascending=[50])

