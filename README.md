## Machine Learning Analysis of International Student Mobility: A Case Study of China
This research project is designed to examine the determinants influencing the mobility of international students, with a specific focus on students from Belt and Road Initiative (B&R) countries studying in China. The objective is to analyze trends in the mobility of international students in China over the past two decades, identify the influencing factors, and forecast future trends.

#### Theoretical Framework: Push-Pull Theory
The study is grounded in the Push-Pull theory, a fundamental framework that elucidates the dynamics of population migration. Ernst G. Ravenstein, in his seminal work "The Law of Migration," articulated that migration factors are dichotomously categorized as "push" factors, which are the repulsive forces in the origin country, and "pull" factors, which are the attractive forces in the destination country. These factors collectively facilitate and drive population movements. This research will employ the Push-Pull theory to categorize and analyze the determinants affecting international student mobility.

#### Methodology and Data Collection
The research will identify push factors including the gross tertiary enrollment ratio, tertiary student-teacher ratio, the ratio of per capita financial expenditure on higher education to per capita GDP, educational inequality, GDP per capita, and the density of R&D researchers. Pull factors will encompass opportunities for scholarships, degree recognition between countries, bilateral trade volume, with geographical distance and the number of outbound students from the origin country serving as control variables.

Data on the number of international students in China and scholarship opportunities will be extracted from the Concise Statistics of International Students Coming to China. Additional data, such as gross tertiary enrollment ratios, student-teacher ratios, and outbound student numbers, will be sourced from the UNESCO database. GDP per capita and R&D researcher metrics will be retrieved from the World Bank, while trade volumes and geographical distances will be obtained from UN Comtrade and the CEPII Database, respectively.

This study will also incorporate web scraping techniques to gather data on the list of Belt and Road countries and the mutual recognition of degrees. Data wrangling methods will be employed to manage and analyze the data, and predictive modeling will be utilized to forecast trends in international student mobility in China.

#### Trends
![inbound based on countries](https://user-images.githubusercontent.com/89746479/215656074-6f7a7fb9-ea67-4dfc-a962-490763f0ac5d.png)

![inbound trend of different study level](https://user-images.githubusercontent.com/89746479/210922241-99b83b8e-8151-4b32-a3f0-95fff0e506d9.png)

![inbound trend of different continents](https://user-images.githubusercontent.com/89746479/210922281-d30c3805-f7c3-462e-ba49-e933aa26853c.png)

![inbound based on countires of different HDI level](https://user-images.githubusercontent.com/89746479/210922298-15f852dc-b744-4078-a2fe-3214e779c940.png)


#### Evaluation
Reduction in AUC ROC 
![reduction in AUC ROC](https://user-images.githubusercontent.com/89746479/211184030-1e07082b-2bc8-475a-828f-b59fe4bd55d2.png)

Partical dependence chart 
![partial dependence](https://user-images.githubusercontent.com/89746479/211183950-b16379da-bb50-40ab-8e5c-61f8ffe32921.png)

![partial dependence2](https://user-images.githubusercontent.com/89746479/211184011-7352a9a8-4ef2-4d76-a864-7286f4f98735.png)


#### Results
The initial phase of the research involved executing multiple models, focusing on data from Asian and African countries due to their data richness. The preliminary results, indicated by high Mean Squared Error (MSE) values, suggested that the selected features were insufficiently predictive within the analytical framework. The homogeneity within continental datasets—reflecting similarities in geographical proximity to China and socio-economic conditions—limited the model’s learning potential. To enhance the model’s predictive capability, I subsequently integrated all available observations into the model training process.

Among various models tested, the K-Nearest Neighbors (KNN) model, with a specification of 10 neighbors, emerged as the most effective. This model, when applied to the test dataset, achieved an R-squared value of 0.79 and an MSE of 0.682. This indicates a substantial degree of accuracy, with an average deviation of 0.682 units between the predicted and actual values, underscoring the effectiveness of the selected variables in forecasting the influx of international students to China.

The Random Forest classifier, with a maximum depth of 6, stood out in the classification model category. This model demonstrated excellent performance on the test data, achieving an accuracy score of 0.97 and a Receiver Operating Characteristic Area Under the Curve (ROC AUC) score of 0.99. An analysis of the feature importance, as illustrated in Figure 6, revealed that the percentage of scholarships, the number of outbound students from a given country, and bilateral trade value were pivotal in predicting the dependent variable. Notably, the scholarship percentage exhibited a significant decrease in influence within the 0.2 to 0.4 range, whereas the impact of the number of outbound students notably increased within the 8 to 8.8 range. In comparison, the student-teacher ratio’s influence remained relatively stable. Figure 8 highlighted a synergistic interaction among these three key predictors, enhancing the model’s predictive power for international student mobility to China.



