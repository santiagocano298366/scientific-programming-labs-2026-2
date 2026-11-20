"""Statistics by group and class separability."""
import pandas as pd
from scipy.stats import shapiro, f_oneway, kruskal
from sklearn.metrics import roc_auc_score


def summary_by_group(dataset, features, groups):
    """Summary of each group."""
    by_group = dataset.groupby("texture_group")[features]
    summary = {"mean": by_group.mean().loc[groups].T,
               "median": by_group.median().loc[groups].T,
               "variance": by_group.var().loc[groups].T}
    return summary


def find_outliers(dataset, features, groups):
    """Outliers of each group."""
    is_outlier = pd.DataFrame(False, index=dataset.index, columns=features)
    for group in groups:
        in_group = dataset["texture_group"] == group
        values = dataset.loc[in_group, features]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        is_outlier.loc[in_group] = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return is_outlier


def test_across_groups(dataset, features, groups):
    """Tests across the groups."""
    alpha = 0.05
    alpha_bonferroni = alpha / len(features)
    results = {"min Shapiro p": [], "test used": [], "statistic": [], "p-value": [], "significant": []}
    for feature in features:
        samples = []
        shapiro_p = []
        for group in groups:
            sample = dataset.loc[dataset["texture_group"] == group, feature]
            samples.append(sample)
            shapiro_p.append(shapiro(sample).pvalue)

        if min(shapiro_p) < alpha:
            statistic, p_value = kruskal(*samples)
            name = "Kruskal-Wallis"
        else:
            statistic, p_value = f_oneway(*samples)
            name = "ANOVA"

        results["min Shapiro p"].append(min(shapiro_p))
        results["test used"].append(name)
        results["statistic"].append(statistic)
        results["p-value"].append(p_value)
        results["significant"].append(p_value < alpha_bonferroni)
    return pd.DataFrame(results, index=features), alpha_bonferroni


def fisher_ratio(values, labels, groups):
    """Fisher ratio of a feature."""
    overall_mean = values.mean()
    between = 0
    within = 0
    for group in groups:
        group_values = values[labels == group]
        between += len(group_values) * (group_values.mean() - overall_mean) ** 2
        within += len(group_values) * group_values.var(ddof=0)
    return between / within


def auc_one_vs_rest(values, labels, group):
    """AUC of one group against the rest."""
    auc = roc_auc_score((labels == group).astype(int), values)
    return max(auc, 1 - auc)


def separability_table(dataset, features, groups):
    """Separability of each feature."""
    labels = dataset["texture_group"]
    table = pd.DataFrame(index=features)
    table["FDR"] = [fisher_ratio(dataset[f], labels, groups) for f in features]
    for group in groups:
        table["AUC " + group + " vs rest"] = [auc_one_vs_rest(dataset[f], labels, group) for f in features]
    return table
