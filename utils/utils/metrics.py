from sklearn.metrics import f1_score

def calc_mean_f1(y_true, y_pred,
                 labels=['Bebê',
                         'Bijuterias e Jóias',
                         'Decoração',
                         'Lembrancinhas',
                         'Papel e Cia']):
    ''' Calculate mean f1 score between labels
    not weighted by proportions.
    '''

    result = f1_score(y_true, y_pred,
             labels=labels,
             average='macro')

    return round(result, 4)