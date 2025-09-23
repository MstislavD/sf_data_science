import numpy as np #для матричных вычислений
import matplotlib.pyplot as plt #для визуализации

from sklearn import model_selection #методы разделения и валидации
from sklearn import metrics

def plot_learning_curve(model, X, y, cv, scoring="f1", ax=None, title=""):
    # Вычисляем координаты для построения кривой обучения
    train_sizes, train_scores, valid_scores = model_selection.learning_curve(
        estimator=model,  # модель
        X=X,  # матрица наблюдений X
        y=y,  # вектор ответов y
        cv=cv,  # кросс-валидатор
        scoring=scoring,  # метрика
    )
    # Вычисляем среднее значение по фолдам для каждого набора данных
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    # Если координатной плоскости не было передано, создаём новую
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))  # фигура + координатная плоскость
    # Строим кривую обучения по метрикам на тренировочных фолдах
    ax.plot(train_sizes, train_scores_mean, label="Train")
    # Строим кривую обучения по метрикам на валидационных фолдах
    ax.plot(train_sizes, valid_scores_mean, label="Valid")
    # Даём название графику и подписи осям
    ax.set_title("Learning curve: {}".format(title))
    ax.set_xlabel("Train data size")
    ax.set_ylabel("Score")
    # Устанавливаем отметки по оси абсцисс
    ax.xaxis.set_ticks(train_sizes)
    # Устанавливаем диапазон оси ординат
    ax.set_ylim(0, 1)
    # Отображаем легенду
    ax.legend()
    
def plot_pr_curve(model, X, y, cv):    
    y_cv_proba_pred = model_selection.cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    y_cv_proba_pred = y_cv_proba_pred[:, 1]
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_cv_proba_pred)
    
    #Вычисляем F1-меру при различных threshold
    f1_scores = (2 * precision * recall) / (precision + recall)
    #Определяем индекс максимума
    idx = np.argmax(f1_scores)
    print('Best threshold = {:.2f}, F1-Score = {:.2f}'.format(thresholds[idx], f1_scores[idx]))

    #Строим PR-кривую
    fig, ax = plt.subplots(figsize=(10, 5)) #фигура + координатная плоскость
    #Строим линейный график зависимости precision от recall
    ax.plot(recall, precision, label='Decision Tree PR')
    #Отмечаем точку максимума F1
    ax.scatter(recall[idx], precision[idx], marker='o', color='black', label='Best F1 score')
    #Даем графику название и подписи осям
    ax.set_title('Precision-recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    #Отображаем легенду
    ax.legend();
    