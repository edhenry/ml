
library(caret)
library(pander)
library(doMC)
library(dplyr)
library(Matrix)
library(plyr)
library(class)
library(d3heatmap)

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

# Create Class for formatting of the TimeStamp field contained within the netflow captures

setClass("myPosixCt")
setAs("character", "myPosixCt", function(from) as.POSIXct(from, format = "%Y/%m/%d %H:%M:%OS"))
options(set.seconds="6")

# Read .binetflow file into dataframe

flowdata_csv <- read.csv("capture20110810.binetflow", colClasses = c("myPosixCt", "numeric", "factor", 
                                                                     "factor","factor","factor","factor",
                                                                     "factor","factor","factor","factor",
                                                                     "numeric", "numeric", "numeric", "factor"), 
                                                                     strip.white = TRUE, sep = ',')

# Subset and normalize data

# Set labels to char for subsetting
flowdata_csv$Label <- as.character(flowdata_csv$Label)

#Function to carve up by timeslice / interval

#flowdata_slice <- timeslice(flowdata_csv, 'secs', 1)

flowdata_slice <- sample_n(flowdata_csv, 75000)

#Define continuous vars, subset flowdata and save as CSV

contvars <- names(flowdata_slice) %in% c("StartTime", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport", "State", "sTos", "dTos")
flowdata_conts <- flowdata_slice[!contvars]

# Normalize the data

cont_vars <- c("Dur", "TotPkts", "TotBytes", "SrcBytes")

#flowdata_conts <- flowdata_conts %>% mutate_each_(funs(normalize), vars = cont_vars)

flowdata_conts[,cont_vars] <- as.data.frame(lapply(flowdata_conts[,cont_vars], normalize))

#flowdata_conts[,cont_vars] <- scale(flowdata_conts[,cont_vars], center = TRUE, scale = TRUE)

#flowdata_conts[,cont_vars] <- log(flowdata_conts[,cont_vars])

# Clean flowdata_conts, totally hacky but dataframe transforms are crazy fast and scale well

flowdata_conts <- flowdata_conts[!(flowdata_conts$Dur == 0),]
flowdata_conts <- flowdata_conts[!(flowdata_conts$TotPkts == 0),]
flowdata_conts <- flowdata_conts[!(flowdata_conts$TotBytes == 0),]
flowdata_conts <- flowdata_conts[!(flowdata_conts$SrcBytes == 0),]

flowdata_conts[flowdata_conts==Inf] <- NA
flowdata_conts[flowdata_conts==-Inf] <-NA
#flowdata_conts[flowdata_conts==0] <- NA
#flowdata_conts[flowdata_conts==NaN] <- NA
#flowdata_conts[flowdata_conts==1] <- NA


flowdata_conts <- na.omit(flowdata_conts)
str(flowdata_conts)

# Re-factor-fy variable

flowdata_conts$Label <- as.factor(flowdata_conts$Label)

# Set randomization seed

set.seed(1234)

# Break dataset into training and test sets
## split dataset randomly with a 67/33% distribution

ind <- sample(2, nrow(flowdata_conts), replace=TRUE, prob=c(0.67, 0.33))

#contvars = c("Dur", "TotPkts", "TotBytes", "SrcBytes")
#labels = c("Label")

flowdata_training <- flowdata_conts[ind==1,]
flowdata_test <- flowdata_conts[ind==2,]

# Display label distribution in datasets

ft_orig <- frqtab(flowdata_conts$Label)
label_freq <- pander(ft_orig, style="rmarkdown", caption="Original Label Frequency (%)")

ft_train <- frqtab(flowdata_training$Label)
ft_test <- frqtab(flowdata_test$Label)
ftcmp_df <- as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ftcmp_df) <- c("Original", "Training Set", "Test Set")
pander(ftcmp_df, style="rmarkdown",
              caption="Comparison of Label frequencies ( in %)")

summary(flowdata_training$Label)

# Define training parameters
    
ctrl <- trainControl(method="repeatedcv", repeats = 1)

# Run training! LET THE COMPUTER OVERLORD LEARN
set.seed(1234)
knnFit1 <- train(Label ~., flowdata_training, method="knn", 
                 trControl = ctrl, tuneLength = 10, na.action = na.omit, 
                 metric = "Accuracy")

#knnFit1 <- knn(train = flowdata_training[,cont_vars], test = flowdata_test[,cont_vars], cl = flowdata_training$Label)

plot(knnFit1)

cmat <- table(flowdata_test$Label, knnFit1)

str(cmat)

heatmap <- d3heatmap(cmat, scale = "column", dendrogram = "none", color="Blues")

heatmap

# Run prediction over test dataset
knnPredict1 <- predict(knnFit1, newdata = flowdata_test)

plot(knnPredict1)

# Calculate confusion matric for prediction accuracy
cmat1 <- confusionMatrix(knnPredict1, flowdata_test_labels)

cmat1

# render plot
# we use three different layers
# first we draw tiles and fill color based on percentage of test cases
tile <- ggplot() +
geom_tile(aes(x=Actual, y=Predicted,fill=Percent),data=cmat1, color="black",size=0.1) +
labs(x="Actual",y="Predicted")
tile = tile + 
geom_text(aes(x=Actual,y=Predicted, label=sprintf("%.1f", Percent)),data=cmat1, size=3, colour="black") +
scale_fill_gradient(low="grey",high="red")
 
# lastly we draw diagonal tiles. We use alpha = 0 so as not to hide previous layers but use size=0.3 to highlight border
tile = tile + 
geom_tile(aes(x=Actual,y=Predicted),data=subset(cmat1, as.character(Actual)==as.character(Predicted)), color="black",size=0.3, fill="black", alpha=0) 
 
#render
tile


