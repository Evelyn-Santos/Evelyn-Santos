import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.core.display import display, HTML
from prettytable import PrettyTable



def get_data():
    df=pd.read_csv("slag_dataset.csv")
    return df

def get_object(df):
    obj_columns = df.select_dtypes(include=['object']).columns.tolist()
    return obj_columns

class Objetos:
    def __init__(self):
        pass
    
    def cuenta_objetos(self, df, obj_columns):
        for column in obj_columns:
            print(f'Datos de la columna: {column}')
            print(df[column].value_counts())
            print('\n')


def resumen(df):
    with pd.option_context("display.max_colwidth", 20):
        info = pd.DataFrame()
        info['muestra'] = df.iloc[0]
        info['tipo de dato'] = df.dtypes
        info['porcentaje datos perdidos'] = df.isnull().sum()*100/len(df)
        return info.sort_values('porcentaje datos perdidos', ascending=False)   

    
def tamaño_dataset(df):
    registros= "{:,}".format(df.shape[0])
    campos= "{:,}".format(df.shape[1])
    print ( "La bd tiene "+ str(campos)+ " columnas y " + str(registros) + " registros")
  
    
def select_data(df):
    data=df[['heat_num','num_cga_met','TCM', 'Time_Vac_Vac','Power_ON','Tpo_Aux','Total_min_Demoras','tco_kwh', 
            'Potencia','O2','Grafito','Cal_Total','Cal_Total_Sid','Cal_Total_Dol','muestra_temp','muestra_ppmo2',
           'grado','c1_tco_tpo_con','c1_tco_kwh','c1_tco_mw_prom','c2_tco_tpo_con','c2_tco_kwh','c2_tco_mw_prom','c7_tco_tpo_con',
           'c7_tco_kwh','c7_tco_mw_prom','cga_met_1','cga_met_2','cga_met_3']]
    return data


def get_unique(df):
    unicos = df['heat_num'].unique()
    return print( "Se tienen " + str(df.shape[0]-len(unicos)) + " registros con el mismo número de colada")


class Cleaner:
    def __init__(self, df):
        self.df = df
        
    def cuenta_duplicados(self):
        duplicados = self.df.duplicated()
        num_duplicados = duplicados.sum()
        print("Hay {} registros duplicados en el conjunto de datos".format(num_duplicados))
    
    def remueve_duplicados(self):
        nuevo_df = self.df.drop_duplicates()
        print("Se han eliminado los registros duplicados")
        return nuevo_df
    
    def crea_id(self):
        self.df.insert(0, 'id', range(1, len(self.df)+1))
        return self.df
    
    def substituye(self):
        self.df = self.df.replace("#¡VALOR!", np.nan)
        return self.df
    
    def separa(self):
        self.df=self.df['grado'].apply(lambda x: x.split("-")[0]).str.split( expand=True).set_axis(['nuevo_grado', 'Y', 'Z'], axis=1)
        self.df=self.df['nuevo_grado']
        return self.df
    
        
def df_nuevo(df):
    return print ("El nuevo conjunto de datos tiene {:,} registros únicos".format(df.shape[0]) + " y {} columnas".format(df.shape[1]))

   
class Flotante(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in X.select_dtypes(include=['object']):
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].astype(float)
            
        return X

class Quita(BaseEstimator, TransformerMixin):
    def __init__(self, umbral=0.5):
        self.umbral = umbral
        
    def fit(self, X, y=None):
        self.columnas_a_quitar = X.columns[X.isnull().mean() > self.umbral]
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.columnas_a_quitar, axis=1)

    
class Mediana(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                continue
            X[col]=pd.to_numeric(X[col], errors="coerce")                
            if pd.api.types.is_numeric_dtype(X[col]):
                self.medians[col] = X[col].median()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(self.medians[col])
        return X


class ContarNegativos:
    def __init__(self):
        self.negativos = {}

    def cuenta_neg(self, df):
        for col in df.select_dtypes(include='number'):
            cuenta = (df[col] < 0).sum()
            if cuenta>0:
                self.negativos[col]=cuenta
        
        return self.negativos
    
    
class CambiaNegativos(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
                   
    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                 X[col] = X[col].apply(lambda x: abs(x) if x < 0 else x)
        return X
   
    
    
class BoxPlot:
    def __init__(self, df, i_col, figsize=(20, 24)):
        self.df = df
        self.figsize = figsize
        self.col=i_col
    
    def bp(self):
        datos_graficar=self.df.iloc[:, self.col:]
        columnas_numericas = datos_graficar.select_dtypes(include=['float', 'int']).columns.tolist()

        fig, axs = plt.subplots(nrows=11, ncols=10, figsize=self.figsize)
        for i, columna in enumerate(columnas_numericas):
            sns.boxplot(x=self.df[columna], ax=axs[i//10, i%10])
        plt.tight_layout()
        plt.show()
        
        

class ReemplazaOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, umbral):
        self.umbral = umbral
        self.quantiles = None
        self.limite_sup = None
        self.limite_inf = None
    
    def fit(self, X, y=None):
        self.quantiles = X.quantile([0.25, 0.75])
        iqr = self.quantiles.iloc[1] - self.quantiles.iloc[0]
        self.limite_sup = self.quantiles.iloc[1] + self.umbral * iqr
        self.limite_inf = self.quantiles.iloc[0] - self.umbral * iqr
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                superior = self.limite_sup[col]
                inferior = self.limite_inf[col]
                if X[col].skew() > 0:
                    superior += self.umbral * (self.quantiles.iloc[1][col] - self.quantiles.iloc[0][col])
                elif X[col].skew() < 0:
                    inferior -= self.umbral * (self.quantiles.iloc[1][col] - self.quantiles.iloc[0][col])
                X[col] = np.where(X[col] > superior, superior, X[col])
                X[col] = np.where(X[col] < inferior, inferior, X[col])
        return X



class Correlaciones:
    
    def __init__(self, alpha, cmap='coolwarm'):
        self.alpha = alpha
        self.cmap = cmap
        self.corr_df = None
        self.pvals_df = None
        self.significativa = None
        self.df_corr_significativa = None
        self.X = None

    def fit_transform(self, X):
        # Calcula las correlaciones entre las variables
        corr = X.corr()

        # Calcula el valor de p de la prueba de hipótesis para cada correlación
        pvals = np.zeros_like(corr)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if i != j:
                    corr_ij, pval_ij = stats.pearsonr(X.iloc[:, i], X.iloc[:, j])
                    pvals[i, j] = pval_ij

        # Convierte la matriz de correlaciones y pvalues en un dataframe
        self.corr_df = pd.DataFrame(data=corr.values, columns=corr.columns, index=corr.index)
        self.pvals_df = pd.DataFrame(data=pvals, columns=corr.columns, index=corr.index)

        # Selecciona sólo las correlaciones con p-valor menor que alpha
        self.significativa = self.corr_df[(self.pvals_df < self.alpha) & (self.pvals_df > 0)]
        self.df_corr_significativa = self.significativa[self.significativa != 0].dropna(how='all').dropna(how='all', axis=1)
        self.df_corr_significativa.fillna(0, inplace=True) # reemplaza NaN con 0

        # Crea dataframe con las correlaciones significativas
        self.significativa_df = pd.DataFrame(self.df_corr_significativa.unstack().sort_values(kind="quicksort")).reset_index()
        self.significativa_df.columns = ['Variable 1', 'Variable 2', 'Correlación']
        self.significativa_df = self.significativa_df[self.significativa_df['Correlación'] != 0]
        print(self.significativa_df)

        # Almacena el dataframe en la instancia de la clase
        self.X = X

        return self.df_corr_significativa


class Horno:
    
    def __init__(self, df, decimales):
        self.df=df
        self.decimales = decimales
        self.promedio_tiempo = self.df.groupby("nuevo_grado")["Time_Vac_Vac"].mean().round(self.decimales)
        self.promedio_potencia = self.df.groupby("nuevo_grado")["Potencia"].mean().round(self.decimales)
        self.desviacion_estandar_tiempo = self.df.groupby("nuevo_grado")["Time_Vac_Vac"].std().round(self.decimales)
        self.percentil_25_tiempo = self.df.groupby("nuevo_grado")["Time_Vac_Vac"].quantile(0.25).round(self.decimales)
        self.percentil_75_tiempo = self.df.groupby("nuevo_grado")["Time_Vac_Vac"].quantile(0.75).round(self.decimales)
        self.desviacion_estandar_potencia = self.df.groupby("nuevo_grado")["Potencia"].std().round(self.decimales)
        self.percentil_25_potencia = self.df.groupby("nuevo_grado")["Potencia"].quantile(0.25).round(self.decimales)
        self.percentil_75_potencia = self.df.groupby("nuevo_grado")["Potencia"].quantile(0.75).round(self.decimales)
        
         
    def mostrar_tabla(self):
        tabla = PrettyTable()
        tabla.field_names = ["Grado", 
                             "TiempoPromFund", 
                             "DesvestFund", 
                             "p25%Fund",
                             "p75%Fund",
                             "Consumo_prom_potencia",
                             "DesvestPot",
                             "p25%Pot",
                             "p75%Pot"]
        tabla.align["Grado"] = "l"
        tabla.max_width["Grado"] = 15
        tabla.max_width["TiempoPromFund"] = 15
        tabla.max_width["DesvestFund"] = 15
        tabla.max_width["p25%Fund"] = 15
        tabla.max_width["p75%Fund"] = 15
        tabla.max_width["Consumo_prom_potencia"] = 15
        tabla.max_width["DesvestPot"] = 15
        tabla.max_width["p25%Pot"] = 15
        tabla.max_width["p75%Pot"] = 15
        for grados, promedio_tiempo in self.promedio_tiempo.items():
            promedio_potencia = self.promedio_potencia[grados]
            desviacion_estandar_tiempo = self.desviacion_estandar_tiempo[grados]
            percentil_25_tiempo = self.percentil_25_tiempo[grados]
            percentil_75_tiempo = self.percentil_75_tiempo[grados]
            desviacion_estandar_potencia = self.desviacion_estandar_potencia[grados]
            percentil_25_potencia = self.percentil_25_potencia[grados]
            percentil_75_potencia = self.percentil_75_potencia[grados]
            tabla.add_row([int(grados), 
                           "{:.{}f}".format(promedio_tiempo, self.decimales),
                           "{:.{}f}".format(desviacion_estandar_tiempo, self.decimales),
                           "{:.{}f}".format(percentil_25_tiempo, self.decimales),
                           "{:.{}f}".format(percentil_75_tiempo, self.decimales),
                           "{:.{}f}".format(promedio_potencia, self.decimales),
                           "{:.{}f}".format(desviacion_estandar_potencia, self.decimales),
                           "{:.{}f}".format(percentil_25_potencia, self.decimales),
                           "{:.{}f}".format(percentil_75_potencia, self.decimales)])
        return print(tabla)                  



class DataframeSelector:
    
    def __init__(self, df):
        self.df = df
        
    def selecciona(self):
        # Seleccionar los registros que tienen en la columna IB3 valores entre 1.1 y 1.8
        datos = self.df[(self.df['IB3'] >= 1.1) & (self.df['IB3'] <= 1.8)]
        
        # Filtrar los registros resultantes para aquellos con valores de "FeO" entre 16 y 20
        datos_filtrados= datos[(datos['feo'] >= 16) & (datos['feo'] <= 20)]
        
        return datos_filtrados
