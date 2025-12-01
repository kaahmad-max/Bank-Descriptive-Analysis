library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ROCR)
library(corrplot)
library(car)
library(MASS)
library(MLmetrics)
library(ggpubr)
library(gridExtra)

# Load the dataset as a data frame
bank <- read.csv("Downloads/bank-full.csv", header = TRUE, stringsAsFactors = FALSE , sep = ";")
#Convert into df
# View the structure of the dataset
str(bank)
# Summary statistics of the dataset
summary(bank)
# Check for missing values
colSums(is.na(bank))
# Convert categorical variables to factors
bank <- bank %>% mutate_if(is.character, as.factor)
# Check the distribution of the target variable
table(bank$y)
# Visualize the distribution of the target variable
ggplot(bank, aes(x = y)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Target Variable", x = "Subscription to Term Deposit", y = "Count") +
  theme_minimal()
# Visualize the distribution of numerical variables
num_vars <- bank %>% select_if(is.numeric)
num_vars_long <- gather(num_vars, key = "variable", value = "value")
ggplot(num_vars_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Distribution of Numerical Variables", x = "Value", y = "Count") +
  theme_minimal()
# Visualize the relationship between categorical variables and the target variable
cat_vars <- names(bank)[sapply(bank, is.factor) & names(bank) != "y"]
for (var in names(cat_vars)) {
  p <- ggplot(bank, aes_string(x = var, fill = "y")) +
    geom_bar(position = "dodge") +
    labs(title = paste("Distribution of", var, "by Target Variable"), x = var, y = "Count") +
    theme_minimal()
  print(p)
}
# Visualize the relationship between numerical variables and the target variable
for (var in names(num_vars)) {
  p <- ggplot(bank, aes_string(x = "y", y = var, fill = "y")) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", var, "by Target Variable"), x = "Subscription to Term Deposit", y = var) +
    theme_minimal()
  print(p)
}
# Correlation matrix for numerical variables
cor_matrix <- cor(num_vars)
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45, title = "Correlation Matrix of Numerical Variables", mar = c(0,0,1,0))
# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(bank$y, p = 0.8, list = FALSE)
train_data <- bank[train_index, ]
test_data <- bank[-train_index, ]
# Train a logistic regression model
logistic_model <- glm(y ~ ., data = train_data, family = binomial)
summary(logistic_model)
# Make predictions on the test set
logistic_preds <- predict(logistic_model, newdata = test_data, type = "response")
logistic_class <- ifelse(logistic_preds > 0.5, "yes", "no")
# Evaluate the logistic regression model
confusionMatrix(as.factor(logistic_class), test_data$y, positive = "yes")
# Train a random forest model
rf_model <- randomForest(y ~ ., data = train_data, ntree = 100)
print(rf_model)
# Make predictions on the test set
rf_preds <- predict(rf_model, newdata = test_data)
# Evaluate the random forest model
confusionMatrix(rf_preds, test_data$y, positive = "yes")
# Compare model performances
logistic_cm <- confusionMatrix(as.factor(logistic_class), test_data$y, positive = "yes")
rf_cm <- confusionMatrix(rf_preds, test_data$y, positive = "yes")
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(logistic_cm$overall['Accuracy'], rf_cm$overall['Accuracy']),
  Precision = c(logistic_cm$byClass['Precision'], rf_cm$byClass['Precision']),
  Recall = c(logistic_cm$byClass['Recall'], rf_cm$byClass['Recall']),
  F1_Score = c(logistic_cm$byClass['F1'], rf_cm$byClass['F1'])
)
print(model_comparison)
# Visualize model comparison
model_comparison_long <- gather(model_comparison, key = "Metric", value = "Value", -Model)
ggplot(model_comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", x = "Metric", y = "Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")
# use xgboost
library(xgboost)
# Prepare data for xgboost  
train_matrix <- model.matrix(y ~ . - 1, data = train_data)
train_label <- ifelse(train_data$y == "yes", 1, 0)
test_matrix <- model.matrix(y ~ . - 1, data = test_data)
test_label <- ifelse(test_data$y == "yes", 1, 0)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)
# Train xgboost model
xgb_model <- xgboost(data = dtrain, nrounds = 100, objective = "binary:logistic", eval_metric = "logloss", verbose = 0)
# Make predictions on the test set
xgb_preds <- predict(xgb_model, newdata = dtest)
xgb_class <- ifelse(xgb_preds > 0.5, 1, 0)
# Evaluate the xgboost model
xgb_cm <- confusionMatrix(as.factor(xgb_class), as.factor(test_label), positive = "1")
print(xgb_cm)
# Add xgboost to model comparison
model_comparison <- rbind(model_comparison, data.frame(
  Model = "XGBoost",
  Accuracy = xgb_cm$overall['Accuracy'],
  Precision = xgb_cm$byClass['Precision'],
  Recall = xgb_cm$byClass['Recall'],
  F1_Score = xgb_cm$byClass['F1']
))
print(model_comparison)
# Visualize updated model comparison
model_comparison_long <- gather(model_comparison, key = "Metric", value = "Value", -Model)
ggplot(model_comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Updated Model Performance Comparison", x = "Metric", y = "Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

#How did “balance” and “housing” variables affect the user behaviour of subscriptions?
# Visualize the effect of "balance" on subscription behavior
ggplot(bank, aes(x = balance, fill = y)) +
  geom_histogram(bins = 30, position = "dodge") +
  labs(title = "Effect of Balance on Subscription Behavior", x = "Balance", y = "Count") +
  theme_minimal()
# Visualize the effect of "housing" on subscription behavior
ggplot(bank, aes(x = housing, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Effect of Housing on Subscription Behavior", x = "Housing Loan", y = "Count") +
  theme_minimal()
# Combined effect of "balance" and "housing" on subscription behavior
ggplot(bank, aes(x = balance, fill = y)) +
  geom_histogram(bins = 30, position = "dodge") +
  facet_wrap(~ housing) +
  labs(title = "Combined Effect of Balance and Housing on Subscription Behavior", x = "Balance", y = "Count") +
  theme_minimal()
# Summary statistics of balance by housing and subscription
balance_summary <- bank %>%
  group_by(housing, y) %>%
  summarise(mean_balance = mean(balance), median_balance = median(balance), count = n())
print(balance_summary)

# Statistical test to check the significance of balance on subscription
t_test_result <- t.test(balance ~ y, data = bank)
print(t_test_result)
# Chi-squared test to check the significance of housing on subscription
chi_sq_result <- chisq.test(table(bank$housing, bank$y))
print(chi_sq_result)
# Logistic regression to assess the combined effect of balance and housing on subscription
combined_model <- glm(y ~ balance + housing, data = bank, family = binomial)
summary(combined_model)
# Predict probabilities using the combined model
bank$predicted_prob <- predict(combined_model, type = "response")
# Visualize predicted probabilities by balance and housing
ggplot(bank, aes(x = balance, y = predicted_prob, color = housing)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(title = "Predicted Probabilities of Subscription by Balance and Housing", x = "Balance", y = "Predicted Probability of Subscription") +
  theme_minimal()
# Conclusion on the effect of balance and housing on subscription behavior
cat("The analysis indicates that both balance and housing loan status significantly affect the likelihood of a customer subscribing to a term deposit. Higher balances are generally associated with higher subscription rates, and customers with housing")
cat(" loans tend to have different subscription behaviors compared to those without housing loans. The logistic regression model further confirms the significance of these variables in predicting subscription behavior
 loans tend to have different subscription behaviors compared to those without housing loans. The logistic regression model further confirms the significance of these variables in predicting subscription behavior.")

# Create a point plot for balance vs subscription with housing as color
ggplot(bank, aes(x = balance, y = as.numeric(y) - 1, color = housing)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(title = "Balance vs Subscription with Housing Status", x = "Balance", y = "Subscription (0 = no, 1 = yes)") +
  theme_minimal()
# Create a boxplot for balance by subscription and housing status
ggplot(bank, aes(x = y, y = balance, fill = housing)) +
  geom_boxplot() +
  labs(title = "Boxplot of Balance by Subscription and Housing Status", x = "Subscription", y = "Balance") +
  theme_minimal()
# Create density plots for balance by subscription and housing status
ggplot(bank, aes(x = balance, fill = y)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ housing) +
  labs(title = "Density Plot of Balance by Subscription and Housing Status", x = "Balance", y = "Density") +
  theme_minimal()
# Summary statistics of balance by housing and subscription
balance_summary <- bank %>%
  group_by(housing, y) %>%
  summarise(mean_balance = mean(balance), median_balance = median(balance), count = n())
print(balance_summary)
# Logistic regression with interaction between balance and housing
interaction_model <- glm(y ~ balance * housing, data = bank, family = binomial)
summary(interaction_model)
#Perform anova test
anova(interaction_model, test = "Chisq")
bank$interaction_predicted_prob <- predict(interaction_model, type = "response")
# Visualize predicted probabilities by balance and housing with interaction
ggplot(bank, aes(x = balance, y = interaction_predicted_prob, color = housing)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(title = "Predicted Probabilities of Subscription by Balance and Housing with Interaction", x = "Balance", y = "Predicted Probability of Subscription") +
  theme_minimal()
