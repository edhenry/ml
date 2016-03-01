
# Import libraries
library(caret)
library(pander)
library(doMC)
library(plyr)
library(dplyr)
library(Matrix)
library(data.table)
library(stringr)
library(FeatureHashing)
library(ggplot2)
library(d3heatmap)

# Install required packages
#list.of.packages <- c("caret", "pander", "doMC", "plyr"
#                     "dplyr", "Matrix", "data.table",
#                     "stringr")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)


# Register CPU core count
registerDoMC(cores=23)

# Utility function for use with % frequency tables
frqtab <- function(x, caption) {
    round(100*prop.table(table(x)), 3)
}

# Utility function to round values in a list
# but only if they are numeric

round_numeric <- function(lst, decimals=2) {
    lappy(lst, function(x){
        if (is.numeric(x)) {
            x <- round(x, decimals)
        }
        return(x)
    })
}

# Utility function for model comparison

summod <- function(cm, fit) {
    summ <- list(k = fit$finalModel$k,
                metric = fit$metric,
                value = fit$results[fit$resultes$k == fit$finalModel$k, fit$metric],
                TN = cm$table[1,1], # True negatives
                TP = cm$table[2,2], # True positives
                FN = cm$table[1,2], # False negatives
                FP = cm$table[2,1], # False positives
                acc = cm$overall["Accuracy"], 
                sens = cm$byClass["Sensitivity"],
                spec = cm$byClass["Specificity"],
                PPV = cm$byClass["Positive Predicted Value"],
                NPV = cm$byClass["Negative Prediced Value"])
    round_numeric(summ)
}

# Utility function to normalize the data

normalize <- function(x){
    num <- x - min(x)
    denom <- max(x) - min(x)
    return (num/denom)
}

#Function to timeslice the data however user would like

timeslice <- function(df, slice, interval) {
    if (slice == 'secs'){
        df <- subset(df, df$StartTime <= df$StartTime[1] + (interval))
        return(df)
    }
    else if (slice == 'mins'){
        df <- subset(df, df$StartTime <= df$StartTime[1] + (interval * 60))
        return(df)
    }
    else if (slice == 'hours') {
        df <- subset(df, df$StartTime <= df$StartTime[1] + (interval * 3600))
        return(df)
    }
    else if (slice == 'days'){
        df <- subset(df, df$StartTime <= df$StartTime[1] + (interval * 86400))
        return(df)
    }
    else
      error <- print("Please enter a valid time interval.")
      return(error)
}

# Read .binetflow file into dataframe

#flowdata_csv <- read.csv("capture20110810.binetflow", colClasses = c("myPosixCt", "numeric", "character", 
                                                                    #"character","character","character",
                                                                    #"character","character","character",
                                                                    #"character","character","numeric", 
                                                                    #"numeric", "numeric", "character"), 
                                                                    #strip.white = TRUE, sep = ',')

flowdata_csv <- fread("capture20110810.binetflow", colClasses = c("character", "numeric", "character", 
                                                                  "character","character","character",
                                                                  "character","character","character",
                                                                  "character","character","numeric", 
                                                                  "numeric","numeric", "character"), 
                                                                  sep = 'auto')

# Set POSIX formatting for StartTime

options(set.seconds="6")
flowdata_csv$StartTime <- as.POSIXct(flowdata_csv$StartTime, format = "%Y/%m/%d %H:%M:%OS")
    
# Trim leading and trailing whitespace
##TODO

# Subset data

#flowdata_slice <- timeslice(flowdata_csv, 'mins', 9)
set.seed(12345)

flowdata_slice <- sample_n(flowdata_csv, 75000)

# Keep only cat vars
cat_vars <- c("Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "Label")

flowdata_slice <- subset(flowdata_slice, select = cat_vars)

str(flowdata_slice)

# Factorize columns in dataframe

flowdata_slice$Proto <- as.factor(flowdata_slice$Proto)
flowdata_slice$SrcAddr <- as.factor(flowdata_slice$SrcAddr)
flowdata_slice$Sport <- as.factor(flowdata_slice$Sport)
flowdata_slice$DstAddr <- as.factor(flowdata_slice$DstAddr)
flowdata_slice$Dport <- as.factor(flowdata_slice$Dport)
flowdata_slice$Label <- as.factor(flowdata_slice$Label)

# Clean get rid of NA's

flowdata_slice <- na.omit(flowdata_slice)

# Set randomization seed

set.seed(1234)

# Break dataset into training and test sets
## split dataset randomly with a 67/33% distribution

ind <- sample(2, nrow(flowdata_slice), replace=TRUE, prob=c(0.67, 0.33))

flowdata_training <- flowdata_slice[ind==1,]
flowdata_test <- flowdata_slice[ind==2,]

# Display label distribution in datasets

ft_orig <- frqtab(flowdata_slice$Label)
label_freq <- pander(ft_orig, style="rmarkdown", caption="Original Label Frequency (%)")

ft_train <- frqtab(flowdata_training$Label)
ft_test <- frqtab(flowdata_test$Label)
ftcmp_df <- as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ftcmp_df) <- c("Original", "Training Set", "Test Set")
pander(ftcmp_df, style="rmarkdown",
              caption="Comparison of Label frequencies ( in %)")

flowdata_test[flowdata_test==""] <- NA
flowdata_training[flowdata_training==''] <- NA

flowdata_training <- as.data.frame(flowdata_training)
flowdata_test <- as.data.frame(flowdata_test)

cat_vars <- c("Proto", "SrcAddr", "Sport", "DstAddr", "Dport")
labels <- c("Label")

flowdata_training_labels <- flowdata_training[,labels]
flowdata_test_labels <- flowdata_test[,labels]

flowdata_training <- flowdata_training[,cat_vars]
flowdata_test <- flowdata_test[,cat_vars]

#dummy_flowdata_training <- dummyVars(~ ., data = flowdata_training, fullRank = TRUE)

#dummy_flowdata_test <- dummyVars(~ ., data = flowdata_test, fullRank = TRUE)

#trsf1 <- data.frame(predict(dummy_flowdata_training, newdata = flowdata_training))
#trfs2 <- data.frame(predict(dummy_flowdata_test, newdata = flowdata_test))

str(flowdata_training)
str(flowdata_training_labels)

str(flowdata_test)
str(flowdata_test_labels)

# Train on the dataset

# Define training parameters
    
ctrl <- trainControl(method="repeatedcv", repeats = 10)

# Run training! LET THE COMPUTER OVERLORD LEARN

flow_model_1 <- train(flowdata_training, flowdata_training_labels, method='nb', trControl=ctrl, tuneLength = 10)

flow_model_1

# Run prediction

flow_model_prediction <- predict(flow_model_1, flowdata_test)

# Calculate confusion matrix for prediction

cmat1 <- confusionMatrix(flow_model_prediction, flowdata_test_labels)
cmat1

str(cmat1$byClass)

cmat1$overall

cmat1_df <- data.frame(cmat1$table$Prediction)


heatmap <- d3heatmap(cmat1, scale = "column", dendrogram = "none", color="Blues")

str(heatmap)


