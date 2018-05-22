##Data Loading##
Test_mmt <- read.csv(file.choose())
Train_mmt <- read.csv(file.choose())

##str of dataset##
str(Test_mmt)
str(Train_mmt)

##Summary of Dataset##
summary(Test_mmt)
summary(Train_mmt)

##Dimension of Dataset##
dim(Test_mmt)
dim(Train_mmt)

##Missing value Treatment##
table(is.na(Train_mmt$B))
Train_mmt$B[is.na(Train_mmt$B)] <- 28.17
Train_mmt$N[is.na(Train_mmt$N)] <- 171.00

Test_mmt$N[is.na(Test_mmt$N)] <- 171.00
Test_mmt$B[is.na(Test_mmt$B)] <- 29.50

##NaN value treatment

table(Train_mmt$A)
Train_mmt$A[Train_mmt$A == " "] <- "NA"
table(is.na(Train_mmt$A))
Train_mmt$A[is.na(Train_mmt$A)] <- "a"

table(Train_mmt$D)
Train_mmt$D[Train_mmt$D == ""] <- "NA"
table(is.na(Train_mmt$D))
Train_mmt$D[is.na(Train_mmt$D)] <- "y"
Train_mmt$D[Train_mmt$D == "l"] <- "y"


table(Train_mmt$E)
Train_mmt$E[Train_mmt$E == ""] <- "NA"
table(is.na(Train_mmt$E))
Train_mmt$E[is.na(Train_mmt$E)] <- "p"
Train_mmt$E[Train_mmt$E == "gg"] <- "p"

table(Train_mmt$G)
Train_mmt$G[Train_mmt$G == ""] <- "NA"
table(is.na(Train_mmt$G))
Train_mmt$G[is.na(Train_mmt$G)] <- "ff"


table(Test_mmt$A)
Test_mmt$A[Test_mmt$A == ""] <- "NA"
table(is.na(Test_mmt$A))
Test_mmt$A[is.na(Test_mmt$A)] <- "a"

table(Test_mmt$D)
Test_mmt$D[Test_mmt$D == ""] <- "NA"
table(is.na(Test_mmt$D))
Test_mmt$D[is.na(Test_mmt$D)] <- "y"

table(Test_mmt$E)
Test_mmt$E[Test_mmt$E == ""] <- "NA"
table(is.na(Test_mmt$E))
Test_mmt$E[is.na(Test_mmt$E)] <- "p"

table(Test_mmt$G)
Test_mmt$G[Test_mmt$G == " "] <- "NA"
table(is.na(Test_mmt$G))
Test_mmt$G[is.na(Test_mmt$G)] <- "ff"

##Boxplot

boxplot(Train_mmt$K)
boxplot(Test_mmt$K)
summary(Test_mmt$O)
summary(Train_mmt$O)

## Outlier treatment ## Treatment has done Only above 3rd Quantile## Below 1st Quantile treatment hasn't been performed
Test_mmt$B[Test_mmt$B > 40 ] <- 29.50 ## Replacing > 3rd quantile values into Median values
Train_mmt$B[Train_mmt$B > 38 ] <- 28.17
Test_mmt$C[Test_mmt$C > 6 ] <- 2.750
Train_mmt$C[Train_mmt$C > 8 ] <- 2.750
Test_mmt$H[Test_mmt$H > 3] <- 0.980
Train_mmt$H[Train_mmt$H > 3] <- 1.0
Test_mmt$K[Test_mmt$K > 3] <- 0
Train_mmt$K[Train_mmt$K > 3] <- 0
Test_mmt$N[Test_mmt$N > 303] <- 171.0
Train_mmt$N[Train_mmt$N > 260] <- 160.0
Test_mmt$O[Test_mmt$O > 500] <- 8.5
Train_mmt$O[Train_mmt$O > 365] <- 3.5



summary(Test_mmt)
summary(Train_mmt)

## Categorical values Treatment ##

Train_mmt$A <- as.numeric(Train_mmt$A)
Test_mmt$A <- as.numeric(Test_mmt$A)
Test_mmt$D <- as.numeric(Test_mmt$D)
Train_mmt$D <- as.numeric(Train_mmt$D)
Test_mmt$E <- as.numeric(Test_mmt$E)
Train_mmt$E <- as.numeric(Train_mmt$E)
Test_mmt$F <- as.numeric(Test_mmt$F)
Train_mmt$F <- as.numeric(Train_mmt$F)
Test_mmt$G <- as.numeric(Test_mmt$G)
Train_mmt$G <- as.numeric(Train_mmt$G)
Test_mmt$I <- as.numeric(Test_mmt$I)
Train_mmt$I <- as.numeric(Train_mmt$I)
Test_mmt$J <- as.numeric(Test_mmt$J)
Train_mmt$J <- as.numeric(Train_mmt$J)
Test_mmt$L <- as.numeric(Test_mmt$L)
Train_mmt$L <- as.numeric(Train_mmt$L)
Test_mmt$M <- as.numeric(Test_mmt$M)
Train_mmt$M <- as.numeric(Train_mmt$M)

str(Test_mmt)
str(Train_mmt)
##Correlation Test
Test_mmt1 = Test_mmt[-1]
Train_mmt1 = Train_mmt[-1]
cortest = cor(Train_mmt1, use="complete.obs", method="spearman")

## Model Building

##1st Iteration
glm_mmt <- glm(P ~., family = binomial(link = "logit"), data = Train_mmt1, control = list(maxit = 50))
summary(glm_mmt)
##2nd Iteration
glm_mmt <- glm(P ~ H + I + J + K + N , family = binomial(link= "logit"), data = Train_mmt1)
summary(glm_mmt) ## AIC 400.21 which is better then the first iteration
pred_mmt1 <- predict(glm_mmt, newdata = subset(Train_mmt1,select = c(8,9,10,11,14)), type = 'response')
pred_mmt1 = ifelse(pred_mmt1 > 0.5,1,0)
pred_mmt1 = as.integer(pred_mmt1)
str(pred_mmt1)
str(Train_mmt1$P)

## Decision tree ##

dec_mmt <- rpart(P~.,data = Train_mmt1, control = rpart.control(cp=0.05,maxdepth = 5,minsplit = 10,
                                                               minbucket = 3))
summary(dec_mmt)
plot(dec_mmt)
rpart.plot(dec_mmt)
pred_dec1 = predict(dec_mmt,Test_mmt1)
pred_dec1 = ifelse(pred_dec1 > 0.5,1,0)
pred_dec1 = as.integer(pred_dec1)
str(pred_dec1)
confusionMatrix(pred_dec1,Train_mmt1$P)

Test_mmt$P = pred_dec1
Submission = subset(Test_mmt,select = c(1,17))

##Random Forest ##
set.seed(415)
Ran_mmt <- randomForest(as.factor(P) ~ .,
                   data=Train_mmt1, importance=TRUE, ntree=2000)
varImp(Ran_mmt) ## F,H,I,J,N.O

Ran_mmt1 <- randomForest(as.factor(P) ~ F + H + I + J + N + O,
                        data=Train_mmt1, importance=TRUE, ntree=2000)
varImpPlot(Ran_mmt)

pred_ran = predict(Ran_mmt,Test_mmt1)
Test_mmt$P = pred_ran
Submission1 = subset(Test_mmt,select = c(1,17))

pred_ran1 = predict(Ran_mmt1,Test_mmt1)
Test_mmt$P = pred_ran1
Submission2 = subset(Test_mmt,select = c(1,17))

write.csv(Submission2, file = "submission2.csv") 
##confusionMatrix(Train_mmt1$P,pred_mmt1)


##Target data class balance check##
table(Train_mmt$H)