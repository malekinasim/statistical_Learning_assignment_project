install.packages(c("tidyverse", "GGally", "car", "corrplot", "fastDummies", "broom"))
library(tidyverse)
library(ggplot2)
library(GGally)
library(car)
library(corrplot)
library(fastDummies)
library(broom)

# خواندن داده
df <- read.csv("E:/myCourseData/staticalLearning/Assignment/project/data/merged_data.csv")

# نمودارهای اولیه
ggplot(df, aes(x = Year, y = Response_Rate, color = Municipality)) +
  geom_line() +
  geom_point() +
  facet_wrap(~Municipality, scales = "free_y") +
  theme_minimal()

df %>%
  group_by(Year) %>%
  summarise(Response_Rate = mean(Response_Rate, na.rm = TRUE)) %>%
  ggplot(aes(x = Year, y = Response_Rate)) +
  geom_line(color = "steelblue") +
  geom_point() +
  labs(title = "Average Math Grade Rate Result Over Time (2015–2023)")

# ماتریس همبستگی و نمودارها
numeric_df <- df %>% select(where(is.numeric))
corr_matrix <- cor(numeric_df, use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8, tl.col = "black", addCoef.col = "black")
ggpairs(numeric_df)

# حذف ویژگی‌های با همبستگی بالا
drop_high_corr_features <- function(data, threshold = 0.7) {
  corr_matrix <- abs(cor(data, use = "complete.obs"))
  drop <- c()
  for (i in 1:(ncol(corr_matrix) - 1)) {
    for (j in (i + 1):ncol(corr_matrix)) {
      if (corr_matrix[i, j] > threshold) {
        col1 <- colnames(corr_matrix)[i]
        col2 <- colnames(corr_matrix)[j]
        if (!col1 %in% drop && !col2 %in% drop &&
            col1 != "Response_Rate" && col2 != "Response_Rate") {
          message(paste("Dropping", col2, "because of high correlation with", col1, ":", corr_matrix[i, j]))
          drop <- c(drop, col2)
        }
      }
    }
  }
  kept <- setdiff(colnames(data), drop)
  list(kept = kept, dropped = drop)
}

corr_result <- drop_high_corr_features(numeric_df, 0.6)
kept_features <- corr_result$kept
dropped_features <- corr_result$dropped

# بررسی VIF
check_vif <- function(df, threshold = 5) {
  df <- df %>% drop_na()
  drop <- c()
  repeat {
    model <- lm(Response_Rate ~ ., data = df)
    vif_values <- vif(model)
    max_vif <- max(vif_values, na.rm = TRUE)
    if (max_vif > threshold) {
      drop_feature <- names(which.max(vif_values))
      message(paste("Dropping", drop_feature, "with VIF =", round(max_vif, 2)))
      df <- df %>% select(-all_of(drop_feature))
      drop <- c(drop, drop_feature)
    } else {
      break
    }
  }
  drop
}

vif_drops <- check_vif(df %>% select(all_of(kept_features), Response_Rate))
dropped_features <- union(dropped_features, vif_drops)
selected_features <- setdiff(names(df), union(dropped_features, c("Municipality", "Year")))


# اصلاح نام‌های ستون‌های دیتافریم
names(df) <- make.names(names(df), unique = TRUE)
names(df)

# حالا می‌توانید مدل‌ها را بدون مشکل اجرا کنید
df_dummies <- df[, c(selected_features, "Municipality", "Year", "Response_Rate")] %>%
  dummy_cols(select_columns = c("Municipality", "Year"), 
             remove_first_dummy = TRUE, remove_selected_columns = TRUE)
names(df_dummies)

colnames(df_dummies) <- gsub("[-[:space:]]", "_", colnames(df_dummies))

ols_backward_elimination <- function(X, y, sl = 0.25) {
  names(X)
  data <- cbind(X, y)
  formula <- as.formula(paste("y ~", paste(names(X), collapse = " + ")))
  model <- lm(formula, data = data)
  
  while (TRUE) {
    pvals <- summary(model)$coefficients[-1, 4]
    if (length(pvals) == 0 || max(pvals) < sl) break
    worst_p <- which.max(pvals)
    feature <- names(pvals)[worst_p]
    
    # بررسی وجود ویژگی قبل از حذف
    if (feature %in% colnames(data)) {
      new_formula <- update(formula(model), paste(". ~ . -", feature))
      new_model <- lm(new_formula, data = data)
      
      if (summary(new_model)$adj.r.squared >= summary(model)$adj.r.squared) {
        message(paste("Dropping", feature, "with p-value =", round(pvals[worst_p], 4)))
        data <- data[, !colnames(data) %in% feature]  # حذف ویژگی
        model <- new_model
      } else {
        message(paste("Keeping", feature, "with p-value =", round(pvals[worst_p], 4)))
        break
      }
    } else {
      break
    }
  }
  model
}


names(df_dummies)
# سپس ادامه فرآیند مدل‌سازی با متغیرهای صحیح
X <- df_dummies %>% select(-Response_Rate,-Response_Rate.1)

names(X)
y <- df_dummies$Response_Rate
names(y)
final_model <- ols_backward_elimination(X, y)
summary(final_model)

