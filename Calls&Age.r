library(tidyverse)
library(janitor)
library(forcats)
library(broom)
library(pROC)
library(margins)
library(rpart)
library(rpart.plot)
library(factoextra)

# 1) Load data
bank <- read.csv("Downloads/bank-full.csv", sep = ";") %>% clean_names()

# 2) Basic transformations
bank <- bank %>%
  mutate(
    y       = factor(y, levels = c("no","yes")),
    y_num   = ifelse(y == "yes", 1, 0),
    age_group = case_when(
      age < 30              ~ "Young",
      age >= 30 & age < 50  ~ "Middle-aged",
      TRUE                  ~ "Senior"
    ),
    age_group = factor(age_group, levels = c("Young","Middle-aged","Senior")),
    # Cap campaign to avoid crazy high values distorting models
    campaign_c = pmin(campaign, 10L),
    contact  = factor(contact),
    poutcome = fct_explicit_na(factor(poutcome), "unknown"),
    job      = factor(job),
    education = factor(education),
    marital   = factor(marital),
    housing   = factor(housing),
    loan      = factor(loan)
  )

# SUMMARY TABLE: calls and success by age group
summary_calls <- bank %>%
  group_by(age_group) %>%
  summarise(
    n = n(),
    avg_calls = mean(campaign),
    success_rate = mean(y_num),
    avg_calls_success = mean(campaign[y == "yes"])
  )
print(summary_calls)

# BOXPLOT: calls by age group
p_box <- ggplot(bank, aes(x = age_group, y = campaign, fill = age_group)) +
  geom_boxplot() +
  labs(title = "Number of Calls by Age Group",
       x = "Age Group", y = "Number of Calls in Campaign") +
  theme_minimal()

# FACET HISTOGRAM: success rate by calls across age groups
p_hist <- ggplot(bank, aes(x = campaign_c, fill = y)) +
  geom_histogram(binwidth = 1, position = "fill") +
  facet_wrap(~ age_group) +
  labs(title = "Success Proportion by Calls and Age Group",
       x = "Number of Calls (capped at 10)", y = "Proportion (Yes/No)") +
  theme_minimal()

p_box
p_hist

# ANOVA: do age groups differ in # calls?
anova_model <- aov(campaign ~ age_group, data = bank)
summary(anova_model)

# BASE model: calls + age_group + interaction
logit_model <- glm(y_num ~ campaign_c * age_group,
                   data = bank, family = binomial())

summary(logit_model)

# Get odds ratios table
or_table <- broom::tidy(logit_model, exponentiate = TRUE, conf.int = TRUE)

or_table <- or_table %>%
  dplyr::select(term, estimate, conf.low, conf.high, p.value)

print(or_table)

# MARGINAL EFFECTS: effect of one extra call by age group
ame_by_age <- margins(logit_model, variables = "campaign_c",
                      at = list(age_group = c("Young","Middle-aged","Senior")))
summary(ame_by_age)

# INTERACTION PLOT: predicted success vs calls by age group
newdata <- expand.grid(
  campaign_c = 0:8,
  age_group = levels(bank$age_group)
)
newdata$pred_prob <- predict(logit_model, newdata, type = "response")

p_interact <- ggplot(newdata, aes(x = campaign_c, y = pred_prob, color = age_group)) +
  geom_line(size = 1.2) +
  labs(title = "Predicted Success Probability by Calls and Age Group",
       x = "Number of Calls (capped at 10)", y = "Predicted P(Subscribe)") +
  theme_minimal() +
  theme(legend.position = "bottom")

p_interact

# MODEL PERFORMANCE (ROC/AUC)
prob <- predict(logit_model, type = "response")
roc_obj <- roc(bank$y, prob)
auc(roc_obj)
plot(roc_obj, main = paste0("ROC Curve (AUC = ", round(auc(roc_obj),3), ")"))


#k-means clustering
set.seed(123)
bank_sample <- bank %>% sample_n(5000)   # 5k is enough for clustering

cluster_data <- bank_sample %>%
  dplyr::select(age, balance, campaign = campaign_c, previous, duration) %>%
  drop_na()

cluster_scaled <- scale(cluster_data)

# Choose k=3 for interpretability
km <- kmeans(cluster_scaled, centers = 3, nstart = 25)
bank_sample$cluster <- factor(km$cluster)

# Cluster summary table
clu_summary <- bank_sample %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    avg_age = mean(age),
    avg_calls = mean(campaign),
    avg_balance = mean(balance),
    success_rate = mean(y_num)
  )
print(clu_summary)

# Cluster vs age_group mix
cluster_age_mix <- bank_sample %>%
  group_by(cluster, age_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(cluster) %>%
  mutate(pct = n / sum(n)) %>%
  arrange(cluster, desc(pct))
print(cluster_age_mix)

# Simple cluster visualisation (first 2 PCs)
fviz_cluster(km, data = cluster_scaled,
             geom = "point", ellipse.type = "norm",
             ggtheme = theme_minimal())

# Decision tree – interpretable rules using calls and age
set.seed(42)
tree_data <- bank %>%
  dplyr::select(y, age, age_group, job, marital, education,
         balance, campaign = campaign_c, previous, poutcome, contact)

tree_model <- rpart(y ~ age_group + age + campaign + job + marital + education +
                      balance + previous + poutcome + contact,
                    data = tree_data,
                    method = "class",
                    control = rpart.control(cp = 0.005, minsplit = 200))

rpart.plot(tree_model, type = 3, extra = 104, fallen.leaves = TRUE,
           main = "Decision Tree – Campaign Success")

# Confusion matrix and accuracy
tree_pred <- predict(tree_model, type = "class")
cm <- table(Predicted = tree_pred, Actual = tree_data$y)
cm
acc <- sum(diag(cm)) / sum(cm)
acc

# Tree-based predicted probabilities for a grid of calls by age_group
grid <- expand.grid(
  campaign = 0:8,
  age_group = levels(bank$age_group),
  age = c(25, 40, 60),  # representatives
  job = names(sort(table(bank$job), TRUE))[1],
  marital = names(sort(table(bank$marital), TRUE))[1],
  education = names(sort(table(bank$education), TRUE))[1],
  balance = median(bank$balance, na.rm = TRUE),
  previous = median(bank$previous, na.rm = TRUE),
  poutcome = names(sort(table(bank$poutcome), TRUE))[1],
  contact = names(sort(table(bank$contact), TRUE))[1]
)

grid$pred_tree <- predict(tree_model, newdata = grid, type = "prob")[,"yes"]

p_tree_interact <- ggplot(grid, aes(x = campaign, y = pred_tree, color = age_group)) +
  geom_line(size = 1.2) +
  labs(title = "Decision Tree – P(Subscribe) vs Calls by Age Group",
       x = "Number of Calls (capped at 10)", y = "Predicted P(Subscribe)") +
  theme_minimal() +
  theme(legend.position = "bottom")

p_tree_interact

