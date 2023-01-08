## Study of Influencing Factors of International Student Mobility--A case of China 

This project aims to study the influence factors of international students' mobility with the case of international students from B&R countries studying in China. The project will study the trend of international students in China in the past two decades. It aims to analysis the influence factors of international students studying in China and predict its trend.

#### Push-Pull Theory
This study will base on Push-Pull theory. The Push-Pull theory is the basic theory that explains population migration. In paper entitled "The Law of Migration", Ernst G. Ravenstein, pointed out that the factors affecting migration can be summarized as "push" and "pull" factors. The "push" factor refers to the exclusionary forces in the country of origin (country) that are not conducive to survival and development, while the "pull" factor refers to the attractiveness of the destination (country). Both factors are forces that contribute to population migration and are the driving force behind population movement. Based on the hypothesis of Push-Pull theory, this study selects factors of two kinds.

#### Data 
The push factors of the study including Gross tertiary enrollment ratio, tertiary student-teacher ratio, and Ratio of per capita financial expenditure on higher education to per capita GDP, the education inequality status of the country, GDP per capita, the number of R&D researchers (per million people). And the pull factors of the study including: the chance for international students to get scholarship, mutual recognition of degrees between the two countries, the total trade volume between two countries. The geographical distance between two countries and the number of outbound students of the origin country are served as control variables in the estimate of international students' mobility.

The dependent variable of number of international students in China and the variable of the chance for international students to get scholarship will be scraped from the Concise Statistics of International Students Coming to China (PDF version). Gross tertiary enrollment ratio, tertiary student-teacher ratio and the number of outbound students of the source country will be downloaded from UNESCO database. GDP per capita, the number of R&D researchers (per million people) will be downloaded from World Bank database. The total trade volume between two countries will be downloaded from UN Comtrade, the geographical distance between two countries will be downloaded from the CEPII Database.

The collection of the list of Belt and Road countries and mutual recognition of degrees will involve scraping technique. In the process of managing and analyzing indicators, data wrangling will be used. The trend of international students' mobility in China will be predicted by machine (statistical) learning.

#### Trends
![inbound trend of different study level](https://user-images.githubusercontent.com/89746479/210922241-99b83b8e-8151-4b32-a3f0-95fff0e506d9.png)

![inbound trend of different continents](https://user-images.githubusercontent.com/89746479/210922281-d30c3805-f7c3-462e-ba49-e933aa26853c.png)

![inbound based on countires of different HDI level](https://user-images.githubusercontent.com/89746479/210922298-15f852dc-b744-4078-a2fe-3214e779c940.png)

#### Evaluation
Reduction in AUC ROC 
![reduction in AUC ROC](https://user-images.githubusercontent.com/89746479/211184030-1e07082b-2bc8-475a-828f-b59fe4bd55d2.png)

Partical dependence chart 
![partial dependence](https://user-images.githubusercontent.com/89746479/211183950-b16379da-bb50-40ab-8e5c-61f8ffe32921.png)

![partial dependence2](https://user-images.githubusercontent.com/89746479/211184011-7352a9a8-4ef2-4d76-a864-7286f4f98735.png)





