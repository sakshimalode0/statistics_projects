head(wine)
str(wine)
summary(wine)
cor(wine)
cov(wine)
var(wine)
#This dataset has several variables, including a dependent variable, wine quality. 
#We can make a linear model and it should work really well because tracking variables is more important than prediction for this kind of dataset.
#We want to know which variable contributes the most to wine quality rather than just predict dependent variable
#Therefore, regression model is appropriate to this dataset.
# The dependent variable is discrete variable.
#It has only integer but it's continuous, not categorical,
#because rational number also has a meaning(for instance, 4.12 isn't integer, but it still works as a quality score).
#Therefore, we could make a linear model without any conversion.
install.packages("QuantPsyc")
library(tidyverse)
library(car)
library(QuantPsyc)
read.csv(file = "C:\\Users\\Dell\\Desktop\\wine.csv", header=T) %>% tibble() -> wine
full_linear_model <- lm(data = wine , formula = quality ~.)
vif(full_linear_model)
#it has multicollinearity. Some variables have too high VIF.
#Let's make a reduced model by following stepwise regression.
#There are three ways in stepwise. Forward, backward and both.
#Normarlly, following backward is better than forward.
reduced_linear_model <- step(object = full_linear_model, direction = "both")
reduced_linear_model$call
#residual plots
plot(reduced_linear_model$residuals)
plot(reduced_linear_model)
#multicollinearity
vif(reduced_linear_model)
mean(vif(reduced_linear_model))
#Normality of residuals
ggplot() + stat_qq(aes(sample = reduced_linear_model$residuals))
summary(reduced_linear_model)
#The summary is up there. R-squared is 0.3567, not bad. t-test says all variables are significant.
confint(reduced_linear_model)
#The confidence interval for all variables doesn't include zero. That means all variables are credible.
lm.beta(reduced_linear_model)
#Standardized regression coefficients are above.
#That says alcohol, volatile acidity and sulphates are more important and pH and free sulfur dioxide aren't.
#In other words, alcohol explains the model best, free sulfur dioxide doesn't.
#CONCLUSION
#We figured out from the result, alcohol is the most powerful variables to red wine quality.
#But it's risky to interpret the result as 'more alcohol, more quality'. We all know that's not true.
#It's appropriate to interpret it as it'd be better to add more alcohol within normal range to might improve wine quality. 
#On the other hand, we can take care less pH and free sulfur dioxide or something.
#Exploratory Data Analysis
plot(wine)
#DECISION TREE
library(rpart)
library(rpart.plot)
m2<- rpart(quality~.,data=wine,method='class')
prp(m2)
#correlation plot for wine data
library(corrgram)
corrgram(wine)
