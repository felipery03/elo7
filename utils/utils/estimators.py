import numpy as np
import pandas as pd
from utils.utils import normalize_text

from sklearn.base import BaseEstimator, TransformerMixin



class FilterColumns(BaseEstimator, TransformerMixin):
    ''' Transformer to filter columns in an input dataframe.
    params:
    col_names(list): List with columns names to filter
    '''
    def __init__(self, col_names=None):
        self.col_names = col_names

    def fit (self, X, y=None):
        return self

    def transform(self, X):
        result = X[self.col_names]
        
        return result

class FillMissing(BaseEstimator, TransformerMixin):
    ''' Fill missing or replace 0 values in weight column with median,
    fill missing concatenated_tag with empty string and 
    replace minimum_quantity from 0 to 1.
    '''    
    def fit(self, X, y=None):
        self.weight_median = X.weight.median()
        
        return self

    def transform(self, X):
        data = X.copy()
        
        data.concatenated_tags.fillna('', inplace=True)
        
        data.loc[data.minimum_quantity==0, 'minimum_quantity'] = 1
        
        data.loc[(data.weight.isnull())|(data.weight==0),
                 'weight'] = self.weight_median
    
        return data
    

class CalcSellerFeatures(BaseEstimator, TransformerMixin):
    ''' Calculate which category the sellers
    already sells to create features bases on seller_id.
    '''    
    def fit(self, X, y):
        data = X.join(y).copy()
        self.seller_cats = data[['seller_id', 'category']].drop_duplicates()
        
        self.seller_cats['value'] = 1
    
        self.seller_cats = self.seller_cats.pivot(index='seller_id',
                                        columns='category',
                                        values='value')

        self.seller_cats.fillna(0, inplace=True)
        
        # Fix column names
        new_col_names = ['seller_cat_' + normalize_text(x).replace(' ', '_') 
             for x in self.seller_cats.columns]

        self.seller_cats.columns = new_col_names

        return self

    def transform(self, X):
        
        data = X.copy()
        data = data.merge(self.seller_cats, how='left',
                      left_on='seller_id',
                      right_index=True)
    
        data.drop('seller_id', axis=1, inplace=True)
        data.fillna(0, inplace=True)
        
        return data

class RecommendSystem_2(BaseEstimator):
    
    def fit(self, X, y=None):

        # Store products info
        self.products = X.join(y).copy()
        self.products.reset_index(drop=True, inplace=True)

        product_cols = ['product_id', 'title', 'category']
        self.products = self.products.groupby(product_cols)[
            ['view_counts', 'order_counts']].max().reset_index()
        
        self.pipe_tfidf = Pipeline([
            ('filter', FilterColumns('title')),
            ('vect', CountVectorizer(tokenizer=tokenize, max_features=1000)),
            ('tfidf', TfidfTransformer())     
        ])

        # Fit tf-idf title transformer
        self.tfidf_matrix = self.pipe_tfidf.fit_transform(self.products, y)
        
        return self
    
    def predict(self, X):
        title = X.copy()
        
        # Calculate tf-idf for title input
        tfidf_string = self.pipe_tfidf.transform(title)
        tfidf_string = tfidf_string.todense()

        # Unpack title tf-idf matrix
        dense_tfidf_matrix = self.tfidf_matrix.todense()

        # Calculate cossine of title input vector and titles matrix
        cos_similarity = np.dot(dense_tfidf_matrix, tfidf_string.T)
        cos_similarity = pd.DataFrame(cos_similarity,
                                      columns=['similarity'])
        
        self.sim_matrix = self.products.join(cos_similarity)
        
        # Predict category
        k = 11
        similar_titles_matrix = self.sim_matrix.sort_values(by='similarity',
                                                            ascending=False)
        
        # Get majoritary class in selected group
        category = similar_titles_matrix.head(k).category.mode()[0]
        
        # Recommend top 10
        top_10 = similar_titles_matrix.query("similarity!=1")
        top_10 = top_10.head(10)[['product_id', 'title']]

        # Pack results in a dict
        top_10 = top_10.product_id.astype('str') + ','+ top_10.title

        result = {'category': category}
        for idx, product in enumerate(top_10):
            result['product_' + str(idx)] = product
                  
        return result