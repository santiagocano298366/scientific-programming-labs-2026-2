# Soil water retention curves: final project

Scientific Computing (MAPC24), Master in Automation and Industrial Control, ITM.
Student: Santiago Cano Molina.

This project organizes the code of Lab 1 and Lab 2 as a pipeline with three modules (preprocessing, analysis and visualization). It also has the complexity analysis of the algorithms and an optimized version of some of them.

## Overview

My thesis wants to estimate the matric potential of the soil with a low-cost sensor and machine learning. The sensor does not exist yet, so here I use a public database of soil water retention curves. The question of the project is: which features of a retention curve change with the soil texture, and how well can three texture groups (coarse, medium and fine) be separated?

## Theoretical background

The soil holds water with a force called matric potential (psi). The relation between psi and the volumetric water content (theta) is the retention curve. The van Genuchten model (1980) describes the curve with four parameters: alpha, n, theta_r and theta_s.

Each curve is described with numbers (features). Some of them are computed with numerical calculus: the area under the curve (trapezoidal rule) and the slope (central difference). To know how well a feature separates the groups I use the Fisher discriminant ratio, the AUC of the ROC curve and the hypothesis tests of the course (Shapiro-Wilk, Kruskal-Wallis and ANOVA, Marusteri and Bacarea 2010).

The cost of the algorithms is written with the Big O notation (Knuth 1976): how the time grows when the size of the input grows.

## Methodology

1. Preprocessing: read the file, remove duplicates, keep the good quality fits, remove missing and impossible values, keep the tropical curves, group the 12 USDA texture classes into three groups and keep the curves with at least 6 points that cover psi <= 1 m and psi >= 100 m.
2. Features: 13 features per curve in three domains (retention level, curve shape and van Genuchten parameters). Each domain is a class with the same method `extract`.
3. Statistics: summary by group, outliers (1.5 IQR), normality and group tests, Fisher ratio and AUC of each feature.
4. Classification: a nearest centroid classifier. The train and test sets are split by site, because the curves of one site are almost equal. The test is repeated with 30 different splits.
5. Complexity: the notebook `complexity_analysis.ipynb` of each module has the analysis line by line, an optimized version and the comparison of times.

## Structure

```
lab3_final_project/
    data/ (raw, processed)
    src/ (preprocessing, analysis, visualization)
    notebooks/ (exploration, reporting)
    results/ (figures, tables)
    tests/
    docs/
    main.py
    requirements.txt, environment.yml, setup.py
```

It follows the example of the assignment. The only addition is `main.py`, a script that runs all the steps with one command. The notebooks do not run the algorithms: they read the files that `main.py` saves.

## How to run

```
pip install -r requirements.txt
python main.py
python tests/test_analysis.py
```

`main.py` reads `data/raw`, saves tables in `data/processed` and `results`, and saves the figures. If the raw file is not in `data/raw`, it is read from GitHub. The notebooks are opened after `main.py` (their folders already have the results).

## Results

* 940 usable curves from 270 sites and 8 references: 538 coarse, 170 medium and 232 fine.
* The 13 features are different between groups (Kruskal-Wallis with Bonferroni correction), but this does not mean that they separate the groups well.
* The best feature is the water content at 100 m (Fisher ratio 1.527, AUC 0.938 for coarse and 0.921 for fine). The three shape features are the weakest.
* Nearest centroid classifier with all the features, tested with sites not used in the training: accuracy 0.78, macro F1 0.711 and macro AUC 0.849. The medium group is the worst (recall 0.500, precision 0.377).
* With 30 different splits and all the features, the accuracy is 0.753 and the macro F1 is 0.705. Always predicting the biggest group gives 0.564.
* The fit of the van Genuchten model takes almost all the time of the features. The ROC curve changed from O(N^2) to O(N log N) and the selection of the curves from O(C R) to O(R).

## Discussion

The water content at high suction is the most useful feature to recognize the texture. The medium group is difficult to separate from the others. Texture and data source are related (one reference gives 53 % of the curves), so a part of the separability can come from the source. Each curve has about ten points, theta_r is not stable (152 of 940 fits end in the lower bound 0) and the p-values are optimistic because the curves of one site are not independent.

## Conclusions

The code of the previous labs is now a pipeline with three modules, documentation and complexity analysis. The three texture groups are not well separated by the features of the curve, so in my thesis the texture should be an input of the model and not something to predict. When I have real data, I need to control the source of the data.

## AI Use Declaration

I used Claude (Anthropic) as a support tool in this project. Specifically: (1) to translate my Spanish drafts into English and polish the wording; (2) to suggest a colorblind-safe color palette for the figures, which I then applied consistently; (3) to help organize the folder structure and the order of the sections, following the directory layout in the assignment statement.

## References

* Gupta, S., Papritz, A., Lehmann, P., Hengl, T., Bonetti, S., & Or, D. (2022). Global Soil Hydraulic Properties dataset based on legacy site and field data. Scientific Data, 9, 444.
* Knuth, D. E. (1976). Big Omicron and big Omega and big Theta. ACM SIGACT News, 8(2), 18-24.
* Marusteri, M., & Bacarea, V. (2010). Comparing groups for statistical differences: how to choose the right statistical test? Biochemia Medica, 20(1), 15-32.
* van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. Soil Science Society of America Journal, 44(5), 892-898.
