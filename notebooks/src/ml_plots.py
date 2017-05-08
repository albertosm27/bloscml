
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
import scoring_functions as sf


def plot_learning_curve(ax, estimator, title, X, y, scoring, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score %s" % scoring.__name__)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.yaxis.grid(color='#CECED2')
    ax.xaxis.grid(color='#CECED2')

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="#FF3B30")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="#4CD964")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#FF3B30",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="#4CD964",
            label="Cross-validation score")

    plt.legend(loc="best")

    return ax


def plot_validation_curve(ax, estimator, X, Y, param_name, param_range,
                          scoring, ylim,  cv):
    train_scores, test_scores = validation_curve(estimator, X, Y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=cv, scoring=scoring,
                                                 n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.set_title("Validation Curve")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Score %s" % scoring.__name__)
    ax.set_ylim(*ylim)
    lw = 2
    ax.yaxis.grid(color='#CECED2')
    ax.xaxis.grid(color='#CECED2')

    ax.plot(param_range, train_scores_mean, label="Training score",
            color="#FF9500", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="#FF9500", lw=lw)
    ax.plot(param_range, test_scores_mean, label="Cross-validation score",
            color="#007AFF", lw=lw)
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="#007AFF", lw=lw)
    plt.legend(loc="best")

    return ax


def plot_nested_cv(non_nested_score, nested_score):
    score_difference = non_nested_score - nested_score

    print('Average difference of %.6f with std. dev. of %.6f.'
          % (score_difference.mean(), score_difference.std()))

    # Plot scores on each trial for nested and non-nested CV
    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    non_nested_scores_line, = plt.plot(non_nested_score, color='#FF3B30')
    nested_line, = plt.plot(nested_score, color='#007AFF')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, 1, 0, 0))

    # Plot bar chart of the difference.
    plt.subplot(122)
    difference_plot = plt.bar(range(20), score_difference, color='#007AFF')
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
               ["Non-Nested CV - Nested CV Score"],
               bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")
    plt.tight_layout()
    plt.suptitle("Non-Nested and Nested Cross Validation",
                 x=.5, y=1.1, fontsize="15")

    return plt


def print_nested_winners(nested_clf, non_nested_clf):
    clf_name = non_nested_clf[0].__class__.__name__
    if (clf_name == 'RandomForestClassifier' or clf_name == 'ExtraTreesClassifier'):
        params = ('criterion', 'bootstrap', 'class_weight')
    elif (clf_name == "MultiOutputClassifier"):
        params = ('estimator__C', 'estimator__gamma')
    else:
        params = ('n_neighbors', 'weights')
    values = [[] for i in range(len(params))]
    for est in non_nested_clf:
        aux_dict = est.get_params()
        for i in range(len(params)):
            values[i].append(aux_dict.get(params[i]))
    print('Non Nested Winners')
    for i in range(len(params)):
        print('%-s --> %-s' % (params[i], Counter(values[i])))
    values = [[] for i in range(len(params))]
    for estimators in nested_clf:
        for est in estimators:
            aux_dict = est.get_params()
            for i in range(len(params)):
                values[i].append(aux_dict.get(params[i]))
    print('Nested Winners')
    for i in range(len(params)):
        print('%-s --> %-s' % (params[i], Counter(values[i])))


class ReportList(list):
    def _repr_html_(self):
        html = ["<table><caption><b>Report</b></caption><thead><tr><th>Name</th>\
        <th>Score</th></tr>"]
        for i, row in enumerate(self):
            html.append("<tr>")
            html.append("<td>%s</td>" % (row[0]))
            if i < 6:
                html.append(
                    "<td>%.4f +/-(%.4f)</td>" % (row[1][0],
                                                 row[1][1]))
            else:
                html.append(
                    "<td>%.4f +/-(%.4f) &nbsp; ~ &nbsp; \
                    %.4f +/-(%.4f)</td>" % (row[1][0],
                                            row[1][1],
                                            row[1][2],
                                            row[1][3]))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


SCORES = [sf.balanced, sf.brier, None, sf.codec, sf.filter_,
          sf.codec_filter, sf.c_level, sf.block, sf.cl_block]
NICE_SCORES = [sf.c_level_nice, sf.block_nice, sf.cl_block_nice]


def cross_val_report(estimator, cv, X, y, no_brier=False):
    report = ReportList()
    for i, scoring in enumerate(SCORES):
        if scoring is None:
            name = 'normal'
            scores = cross_val_score(estimator, cv=cv, X=X, y=y,
                                     scoring=scoring)
            values = [scores.mean(), scores.std()]
        elif no_brier and scoring.__name__ == 'brier':
            values = [0, 0]
            name = scoring.__name__
        else:
            scores = cross_val_score(estimator, cv=cv, X=X, y=y,
                                     scoring=scoring)
            values = [scores.mean(), scores.std()]
            name = scoring.__name__
        if (i > 5):
            scores = cross_val_score(estimator, cv=cv, X=X, y=y,
                                     scoring=NICE_SCORES[i - 6])
            values.extend([scores.mean(), scores.std()])
        report.append([name, values])
    return report
