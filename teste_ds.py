import sys
import pickle

import numpy as np
import pandas as pd
from utils.estimators import RecommendSystem_2


def main():
    if len(sys.argv) == 3:
        # Read input
        mode, input_data = sys.argv[1:]

        # Load models
        CLASSIFIER_PATH = "models/classifier.pkl"
        RECOMMENDER_PATH = "models/recommender.pkl"
        classifier = pickle.load(open(CLASSIFIER_PATH, 'rb'))['model']
        recommender = pickle.load(open(RECOMMENDER_PATH, 'rb'))['model']

        if mode == '--category':

            # Prep input
            vars = eval(input_data)
            vars_df = pd.DataFrame(vars.items())
            vars_df = vars_df.set_index(0).T.reset_index(drop=True)

            # Fix types
            vars_df.price = vars_df.price.astype(float)
            vars_df.weight = vars_df.weight.astype(int)
            vars_df.express_delivery = vars_df.express_delivery.astype(int)
            vars_df.minimum_quantity = vars_df.minimum_quantity.astype(int)

            # Predict Category
            result = classifier.predict(vars_df)[0][0]

            print(result)

        elif mode == '--recommendation':
            # Prep input
            title = pd.DataFrame({'title': [input_data]})
            result = recommender.predict(title)

            for key, value in result.items():
                print(value)

        else:
            print(r'''Por favor, entre um modo de execução válido.
             "--category" ou "--recommendation".''')

    else:
        print('''
            Por favor forneça qual tipo de modelo deve ser executado e a
            informação adequada. Para executar o classificador utilizando
            as características de um produto utilize como primeiro parâmetro
            --category e segundo parâmetro as variáveis no seguinte formato
            {'feature_1':<input_da_feature_1>,'feature_2':<input_da_feature_2>
            ,...}\n
            exemplo: python teste_ds.py --category
            "{'title':'Mandala Espírito Santo',
            'concatenated_tags':'mandala mdf',
            'price': 171.89,'weight': 1200,'express_delivery':1,
            'minimum_quantity': 4}"

            Para utilizar o sistema de recomendação forneça como
            primeiro parâmetro --recomendation e como segundo parâmetro
            um título de algum produto. \n
            exemplo: python teste_ds.py --recommendation "abajur"
            ''')


if __name__ == '__main__':
    main()
