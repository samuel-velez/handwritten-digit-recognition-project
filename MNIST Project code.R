library(dslabs)
mnist <- read_mnist()

## Pre-processing

set.seed(1)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])

set.seed(1)
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(mnist$test$images)

library(matrixStats)
library(ggplot2)

sds <- colSds(x)
qplot(sds,bins = 256)

library(caret)

lowvariance <- nearZeroVar(x)
image(matrix(1:784 %in% lowvariance, 28, 28))

col_ind_x <- setdiff(1:ncol(x),lowvariance)
length(col_ind_x)

## Fitting models
# k-Nearest Neighbors

control_set <- trainControl(method = "cv", number = 10, p = .9)

set.seed(1)

train_knn <- train(x[,col_ind_x], y, 
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(1,9,2)),
                   trControl = control_set)
train_knn
model_knn <- knn3(x[, col_ind_x], y,  k = train_knn$bestTune)

y_hat_knn <- predict(model_knn, x_test[, col_ind_x], type="class")
confusionmatrix <- confusionMatrix(y_hat_knn, factor(y_test))

confusionmatrix$overall["Accuracy"]
confusionmatrix$byClass[,1:2]

p_max <- predict(model_knn, x_test[,col_ind_x])
p_max <- apply(p_max, 1, max)
ind <- which(y_hat_knn != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]
#image(matrix(ind, 28, 28))

for(i in ind[1:6]){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste0("We guessed Pr(",y_hat_knn[i],")=",round(p_max[i], 2), " but it is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

# Random Forests

library(Rborist)

control_set <- trainControl(method="cv", number = 5, p = 0.8)
tunegrid <- expand.grid(minNode = c(1) , predFixed = seq(10, 40, 5))

train_randforest <-  train(x[, col_ind_x], 
                           y, 
                           method = "Rborist", 
                           nTree = 50,
                           trControl = control_set,
                           tuneGrid = tunegrid,
                           nSamp = 5000)

ggplot(train_randforest)
train_randforest$bestTune

model_randforest <- Rborist(x[ ,col_ind_x], y, 
                            nTree = 1000,
                            minNode = train_randforest$bestTune$minNode,
                            predFixed = train_randforest$bestTune$predFixed)

y_hat_rf <- factor(levels(y)[predict(model_randforest, x_test[ ,col_ind_x])$yPred])

confusionmatrix <- confusionMatrix(y_hat_rf, y_test)

confusionmatrix$overall["Accuracy"]
confusionmatrix$byClass[,1:2]

p_max <- predict(model_randforest, x_test[,col_ind_x])$census  
p_max <- p_max / rowSums(p_max)
p_max <- apply(p_max, 1, max)
ind  <- which(y_hat_rf != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]

for(i in ind[1:6]){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste0("We guessed Pr(",y_hat_rf[i],")=",round(p_max[i], 2), " but it is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

# Ensemble

p_randforest <- predict(model_randforest, x_test[,col_ind_x])$census 
p_randforest <- p_randforest / rowSums(p_randforest)
p_knn  <- predict(model_knn, x_test[,col_ind_x])
p <- (p_randforest + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)

confusionmatrix <- confusionMatrix(y_pred, y_test)
confusionmatrix$overall["Accuracy"]
confusionmatrix$byClass[,1:2]