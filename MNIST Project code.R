##Introduction
#The MNIST ( Modified National Institute of Standards and Technology) database is a compilation of handwritten numbers. Each of the components in the data set include a matrix with features in the columns, which we will use to train a couple models to find which one performs best recognizing the handwritten numbers.
#Each entry is composed of 784 features, which represent each pixel in a 78x78 image, and their values depend on the "intensity" of the pixel, ranging between 0 to 255, 0 being white and 255 black.

#The library is free to access through the dslabs package:

library(dslabs)
mnist <- read_mnist()

#The data set is already partitioned into training and test sets, with 60.000 and 10.000 entries respectively.

dim(mnist$test$images)
dim(mnist$train$images)

#Because this default data set contains so many entries (more than my computer can handle), I will reduce it by taking 10.000 random samples to train our model and 1.000 to test it.

set.seed(123)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])

#Column names are added:
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(mnist$test$images)


##Pre-processing

#Since the digits are all centered, some spaces in the 78x78 images may go entirely unused, so it can be suspected that some features are useless, and could be removed. This will be tested by checking their variability, and those with zero or near zero variability will be deleted:

library(matrixStats)
sds <- colSds(x)
library(ggplot2)
qplot(sds,bins = 256)

#This plot shows that some parts of the image never change. The nearZeroVar function can recommend which features should be deleted, by calculating the frequency of the most common values in each feautre.

library(caret)
lowvariance <- nearZeroVar(x)

#Plotting the features that can be removed (the yellow ones being the ones that will be kept):

image(matrix(1:784 %in% lowvariance, 28, 28))
#This reduces our number of columns from 784 to 252

col_ind_x <- setdiff(1:ncol(x),lowvariance)
length(col_ind_x)


##Fitting models

#The models that will be tested are k-Nearest Neighbors and Random Forest, as they are both relatively simple to fit, yet provide the necessary flexibility.

##k-Nearest Neighbors
#kNN models work by using the "k" nearest data points to the observed one to estimate its conditional probability. The advantage of applying k Nearest Neighbors is how responsive it is to local structures.
#The goal with kNN models is to optimize k to achieve maximum accuracy.

#The following code will find the k that provides the best accuracy, using the k-fold cross validation method with the trainControl function, which will fold our data in 10, and use 10% of each to test itself.
control_set <- trainControl(method = "cv", number = 10, p = .9)
#And now we train the model trying out k from 1 to 9 by 2.
set.seed(123)
train_knn <- train(x[,col_ind_x], y, 
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(1,9,2)),
                   trControl = control_set)
train_knn$bestTune

#The results indicate that the optimal k is 1, so it will be the chosen one to fit the model:
model_knn <- knn3(x[, col_ind_x], y,  k = train_knn$bestTune)

#And now we will apply it to testing data set and get a good conclusion on its performance:

y_hat_knn <- predict(model_knn, x_test[, col_ind_x], type="class")

confusionmatrix <- confusionMatrix(y_hat_knn, factor(y_test))
confusionmatrix$overall["Accuracy"]

#This k provides an accuracy of 0.946 in the test subset.

confusionmatrix$byClass[,1:2]

#Closer inspection of each class reveals that the number with the lowest sensitivity (the one that is most often guessed incorrectly) is 8 with a true positive rate of 88.6%.

##Random Forests
#Random Forest models create a certain amount of decision trees and then creates an ensemble of them. Each tree in each forest is subject to several parameters that can be tuned to maximize precision.
#Using the Rborist package's Random Trees, the parameters that will be optimized are the number of randomly selected predictors (predFixed) to split the "branches" of the trees, and minimal node size (minNode), which defines how many data points have to be in a node to create a "branch".
#The random number of predictors ensures the trees will not be correlated, increasing precision. We wil test values between 10 and 50 skipping by 5.
#Meanwhile the minimal node size defines how deep the tree will be. This last parameter will be fixed at one, to avoid calculating an optimal one yet letting the tree run as deep as it needs to.
#When building each tree, a random subset of observations will be taken using the nSamp function.
#The cross validation method will be used again, but this time folding it only 5 times and with only 50 trees, as processing requirements are more demanding for this model than for kNN.

library(Rborist)
control_set <- trainControl(method="cv", number = 5, p = 0.8)
tunegrid <- expand.grid(minNode = c(1) , predFixed = seq(10, 40, 5))

set.seed(123)
train_randforest <-  train(x[, col_ind_x], 
                   y, 
                   method = "Rborist", 
                   nTree = 50,
                   trControl = control_set,
                   tuneGrid = tunegrid,
                   nSamp = 5000)

ggplot(train_randforest)
train_randforest$bestTune

#This shows that the optimal number of random predictors is 10 while the minimal node size is fixed at 1. Knowing this, the number of trees can be increased to create the actual model.

set.seed(123)
model_randforest <- Rborist(x[ ,col_ind_x], y, 
                  nTree = 1000,
                  minNode = train_randforest$bestTune$minNode,
                  predFixed = train_randforest$bestTune$predFixed)

y_hat_rf <- factor(levels(y)[predict(model_randforest, x_test[ ,col_ind_x])$yPred])

confusionmatrix <- confusionMatrix(y_hat_rf, y_test)
confusionmatrix$overall["Accuracy"]

#The model provides an accuracy of 0.951, slightly worse than our previous knn result.

confusionmatrix$byClass[,1:2]

#Meanwhile, the most problematic number is again the 8, with a true positive rate of 88.6%. This means that the performance of Random Forests was slightly better accuracy-wise and our sensitivity and specificity were also improved.
#To conclude with these two results, the flexibility in Random Forests allowed for a better performance, but not far from knn, which also had a good performance. I suspect that had its processing not being restricted so much, Random Forest could provide an even better result.

#We can make an effort to improve on our previous results by combining the models, which can be done by simply taking the average of the class probabilities for each prediction:

set.seed(123)
p_randforest <- predict(model_randforest, x_test[,col_ind_x])$census 
p_randforest <- p_randforest / rowSums(p_randforest)
p_knn  <- predict(model_knn, x_test[,col_ind_x])
p <- (p_randforest + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)

confusionmatrix <- confusionMatrix(y_pred, y_test)
confusionmatrix$overall["Accuracy"]
confusionmatrix$byClass[,1:2]

#Doing this pushes the accuracy into 0.953, indeed improving our result, but the current best approaches hover over 0.970 accuracy. Our most problematic digits remain the 8 and 5. Improving that will be an interesting challenge.

#Plotting numbers:
p_max <- predict(model_knn, x_test[,col_ind_x])
p_max <- apply(p_max, 1, max)
ind <- which(y_hat_knn != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]
image(matrix(ind, 28, 28))

for(i in ind[1:12]){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste0("Pr(",y_hat_rf[i],")=",round(p_max[i], 2), " but is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

