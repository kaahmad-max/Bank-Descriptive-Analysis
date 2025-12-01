library('tidyverse')
library('forcats')
library('ggplot2')
library('scales')
library('ggtext')
library('rpart')
library('rpart.plot')
library('smotefamily')
library('pROC')
library('caret')


#load data
bank <- read.csv('~/Desktop/wd/bank-full.csv',sep = ";")

train_data$y <- as.factor(train_data$y)
test_data$y  <- as.factor(test_data$y)

# Make sure "yes" is the minority and set as reference
train_data$y <- relevel(train_data$y, ref = "yes")

X_train <- train_data[, setdiff(names(train_data), "y")]
y_train <- train_data$y
# Example: split into train/test (adjust to your split method)
set.seed(123)
idx <- sample(seq_len(nrow(bank)), size = 0.7 * nrow(bank))
train_data <- bank[idx, ]
test_data  <- bank[-idx, ]

# Make outcome a factor
train_data$y <- as.factor(train_data$y)
test_data$y  <- as.factor(test_data$y)

# Set "yes" as reference level (minority)
train_data$y <- relevel(train_data$y, ref = "yes")

# Remove the target column
X_train <- subset(train_data, select = -y)

# Make sure all character columns are factors
char_cols <- sapply(X_train, is.character)
X_train[char_cols] <- lapply(X_train[char_cols], factor)

# One-hot encode all factors -> numeric matrix
X_train_num <- model.matrix(~ . - 1, data = X_train)  # -1 removes intercept
X_train_num <- as.data.frame(X_train_num)

smote_out <- SMOTE(
  X      = X_train_num,
  target = train_data$y   # factor with "yes"/"no"
)

smote_data <- smote_out$data

# smote_out$data has predictors + a column named "class"
colnames(smote_data)[ncol(smote_data)] <- "y"
smote_data$y <- as.factor(smote_data$y)

# Check new class balance
table(smote_data$y)

confusionMatrix(as.factor(logistic_class), test_data$y, positive = "yes")


# AGE distribution by campaign response
mean_age <- mean(bank$age, na.rm = TRUE)
bank %>% group_by(age) %>% summarise(rate = mean(y=="yes"))
print(mean_age)
ggplot(bank, aes(x = age, fill = y)) +
  geom_histogram(position = "identity",
                 alpha = 0.6,
                 bins = 30,
                 color = "black",
                 na.rm = TRUE) +
  geom_vline(aes(xintercept = mean_age),
             linetype = "solid",
             size = 1,
             colour = "grey") +
  scale_fill_manual(values = c("no" = "brown2", "yes" = "darkolivegreen3")) +
  scale_x_continuous(
    breaks = seq(15, max(bank$age, na.rm = TRUE), by = 5),
    limits = c(15, max(bank$age, na.rm = TRUE))
  ) +
  labs(
    title = "Age Distribution by Campaign Response",
    x = "Age",
    y = "Count",
    fill = "Subscribed?",
    subtitle = paste("Mean age =", round(mean_age, 0))
  ) +
  theme_minimal()


#OCCUPATION
job_rates <- bank %>%
  group_by(job) %>%
  summarise(
    sub_rate = mean(y == "yes"),
    n = n()
  ) %>%
  arrange(desc(sub_rate)) %>%
  mutate(
    job = factor(job, levels = job),             # keep sorted order
    top4 = job %in% c("student", "retired", "unemployed", "management")
  )

job_rates
ggplot(job_rates, aes(x = job, y = sub_rate, fill = top4)) +
  geom_col(color = "black") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(
    values = c("TRUE" = "darkolivegreen3", "FALSE" = "grey80"),
    guide = "none"
  ) +
  labs(
    title = "Subscription Rate by Occupation",
    x = "Occupation",
    y = "Subscription rate (Yes %)"
  ) +
  theme_minimal()

#A. Marital status staked bar plot
bank %>%
  group_by(marital, y) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(marital) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = marital, y = prop, fill = y)) +
  geom_col(position = "fill", color = "black") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("no" = "brown2", "yes" = "darkolivegreen3")) +
  labs(
    title = "Subscription Rate by Marital Status",
    x = "Marital Status",
    y = "Subscription Rate",
    fill = "Subscribed?"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold")
  )

#B.Marital status power plot
best_group <- bank %>%
  group_by(marital) %>%
  summarise(sub_rate = mean(y == "yes")) %>%
  arrange(desc(sub_rate)) %>%
  slice(1) %>%
  pull(marital)

highlight_plot <- bank %>%
  group_by(marital) %>%
  summarise(sub_rate = mean(y == "yes")) %>%
  mutate(best = marital == best_group) %>%
  ggplot(aes(x = marital, y = sub_rate)) +
  
  geom_segment(
    aes(x = marital, xend = marital, y = 0, yend = sub_rate,
        color = best, linewidth = best),
    show.legend = FALSE
  ) +
  
  geom_point(
    aes(color = best),
    size = 5,
    show.legend = FALSE
  ) +
  
  scale_color_manual(values = c("FALSE" = "azure4", "TRUE" = "darkolivegreen3")) +
  scale_linewidth_manual(values = c("FALSE" = 1, "TRUE" = 2.5)) +
  
  scale_y_continuous(labels = scales::percent) +
  
  
  labs(
    title = "Subscription Rate by Marital Status",
    subtitle = paste("Best-performing group:", best_group),
    x = NULL,
    y = "Subscription Rate"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = ggtext::element_markdown()
  )

highlight_plot


#C.Marital status histogram
ggplot(bank, aes(x = age, fill = y)) +
  geom_histogram(
    bins = 25,
    alpha = 0.7,
    position = "identity",
    color = "black"
  ) +
  facet_wrap(~ marital) +
  scale_fill_manual(values = c("no" = "brown2", "yes" = "darkolivegreen3")) +
  labs(
    title = "Age Distribution by Marital Status and Response",
    x = "Age",
    y = "Count",
    fill = "Subscribed?"
  ) +
  theme_minimal()

#EDUCATION LEVEL
best_level <- bank %>%
  group_by(education) %>%
  summarise(sub_rate = mean(y == "yes")) %>%
  arrange(desc(sub_rate)) %>%
  slice(1) %>%
  pull(education)

bank %>%
  group_by(education, y) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(education) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = fct_reorder(education, prop, .fun = max), 
             y = prop, fill = y)) +
  geom_col(position = "fill") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("no" = "brown3", "yes" = "darkolivegreen3")) +
  labs(
    title = "Subscription Rate by Education Level",
    subtitle = paste("Highest-performing group:", best_level),
    x = "Education level",
    y = "Percentage of customers",
    fill = "Subscribed?"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(
      face = ifelse(levels(fct_reorder(bank$education, bank$age)) == best_level,
                    "bold", "plain")
    )
  )

bank %>%
  group_by(education) %>%
  summarise(sub_rate = mean(y == "yes")) %>%
  ggplot(aes(x = fct_reorder(education, sub_rate), y = sub_rate)) +
  geom_col(fill = "darkolivegreen3") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Subscription Rate by Education Level",
    x = "Education level",
    y = "Subscription rate (Yes %)"
  ) +
  theme_minimal()

model_comparison_long <- model_comparison %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")


#LOGITIC REGRESSION

# Outcome as factor (yes/no)
bank$y <- as.factor(bank$y)

# Make "no" the reference (so model predicts P(yes))
bank$y <- relevel(bank$y, ref = "no")

############################################################
## 3. Train–test split
############################################################

set.seed(123)
n  <- nrow(bank)
id <- sample(seq_len(n), size = 0.7 * n)

train_data <- bank[id, ]
test_data  <- bank[-id, ]

############################################################
## 4. Select PERSONAL BACKGROUND variables
############################################################

bg_vars <- c("age", "job", "marital", "education",
             "default", "housing", "loan")

# Just to be safe, make sure they exist:
bg_vars <- bg_vars[bg_vars %in% names(bank)]

############################################################
## 5. Create numeric matrices for SMOTE (train only)
############################################################

# Formula for dummy encoding (no intercept)
mm_formula <- as.formula(
  paste("~", paste(bg_vars, collapse = " + "), "-1")
)

# TRAIN predictors (personal background only)
X_train_num <- model.matrix(mm_formula, data = train_data)
X_train_num <- as.data.frame(X_train_num)

# TRAIN target
y_train <- train_data$y

############################################################
## 6. Apply SMOTE on TRAINING DATA
############################################################

smote_out <- SMOTE(
  X      = X_train_num,
  target = y_train,
  K      = 5
)

smote_data <- smote_out$data

# Last column is the class label
colnames(smote_data)[ncol(smote_data)] <- "y"
smote_data$y <- as.factor(smote_data$y)

# Check new balance
cat("Class balance after SMOTE:\n")
print(table(smote_data$y))

############################################################
## 7. Fit logistic regression on SMOTE data
############################################################

logit_smote <- glm(
  y ~ .,
  data   = smote_data,
  family = binomial
)

summary(logit_smote)

# TEST predictors (personal background only)
X_test_num <- model.matrix(mm_formula, data = test_data)
X_test_num <- as.data.frame(X_test_num)

# Make sure columns match training matrix
missing_cols <- setdiff(colnames(X_train_num), colnames(X_test_num))
for (col in missing_cols) {
  X_test_num[[col]] <- 0
}
# Order columns the same as in training
X_test_num <- X_test_num[, colnames(X_train_num)]

# Predicted probabilities of "yes"
test_prob <- predict(logit_smote, newdata = X_test_num, type = "response")

# Choose a cutoff (you can play with this: 0.3, 0.2, etc.)
cutoff <- 0.4917226

test_pred <- ifelse(test_prob >= cutoff, "yes", "no")
test_pred <- factor(test_pred, levels = levels(test_data$y))

cat("\nConfusion matrix on TEST data (cutoff =", cutoff, "):\n")
print(table(Predicted = test_pred, Actual = test_data$y))



#ROC
roc_obj <- roc(test_data$y, test_prob)
plot(roc_obj, print.auc = TRUE, col = "blue")
coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
best_cutoff <- coords(roc_obj, "best", ret = "threshold")
best_cutoff
test_pred_opt <- ifelse(test_prob >= best_cutoff, "yes", "no")
test_pred_opt <- factor(test_pred_opt, levels = levels(test_data$y))

table(Predicted = test_pred_opt, Actual = test_data$y)
library(caret)
length(test_pred_opt)
length(test_data$y)

confusionMatrix(
  data = test_pred_opt,   # your predicted classes
  reference = test_data$y # actual classes
)

test_pred_opt <- ifelse(test_prob >= cutoff, "yes", "no")

# 3. Make it a factor with the same levels as the actual y
test_pred_opt <- factor(test_pred_opt, levels = levels(test_data$y))

# 4. Check lengths now match
length(test_pred_opt)  # should be 13564
length(test_data$y)    # should be 13564


# 6. Print a nice confusion matrix
confusionMatrix(
  data      = test_pred_opt,
  reference = test_data$y
)

test_prob <- predict(logit_smote, newdata = X_test_num, type = "response")
confusionMatrix(
  data = test_pred_opt,
  reference = test_data$y
)
install.packages("knitr")
cm <- table(Predicted = test_pred_opt, Actual = test_data$y)

kable(cm, caption = "Confusion Matrix")
cm <- table(Predicted = test_pred_opt, Actual = test_data$y)
cm_df <- as.data.frame(cm)

# Plot it
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(
    title = "Confusion Matrix",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal(base_size = 14)
