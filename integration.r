##setcol <- c("emp ID",
##            "name",
##            "DesignationName",
##            "Unit_Name",
##            "TeamName",
##            "Active",
##            "marstat",
##          "GENDER"
##          )

Master <- read.csv(file.choose(), header = T, na.strings =c("","NA"),sep = "\t")
summary(Master)
str(Master)
unique(Master[c("Empcode")])

## Loading salary data

Sal_data <- read.csv(file.choose())
summary(Sal_data)
str(Sal_data)
unique(Sal_data[c("EMPLOYEE_CODE")])


## Loading primary dataset

primary <- read.csv(file.choose())
primary <- primary[c(-14,-15,-16)]



##Taking care of Gender from Master dataset
##Master$GENDER[Master$GENDER == 1] <- 'Male'
##Master$GENDER[Master$GENDER == "Hindu"] <- 'Male'
##Master$GENDER[Master$GENDER == 'male'] <- 'Male'
##Master$GENDER[Master$GENDER == 'Male'] <- 'Male'
##Master$GENDER[Master$GENDER == 'MAle'] <- 'Male'
##Master$GENDER[Master$GENDER == 'MALE'] <- 'Male'
##Master$GENDER[Master$GENDER == 'Mlae'] <- 'Male'
##Master$GENDER[Master$GENDER == 'Mr.']  <- 'Male'
##Master$GENDER[Master$GENDER == 2] <- 'Female'
##Master$GENDER[Master$GENDER == 'Famale'] <- 'Female'
##Master$GENDER[Master$GENDER == 'Female'] <- 'Female'
##Master$GENDER[Master$GENDER == 'FEMALE'] <- 'Female'
##Master$GENDER[Master$GENDER == 'Femlae'] <- 'Female'
##Master$GENDER[Master$GENDER == 'Ms.'] <- 'Female'

Master$GENDER[is.na(Master$GENDER)] <- 'Female'
Gender<- Master$GENDER
Gender <- factor(Gender)
levels(Gender) <- list(M=c("Male", "Mr.","Hindu","male", "MAle", "MALE", "Mlae"), 
                    Fe=c("Female", "Famale", "FEMALE","Femlae", "Ms."))
Gender

## Managing Marital status from Master dataset
Master$marstat[is.na(Master$marstat)] <- 'Married'
Marital.Status<- Master$marstat
Marital.Status <- factor(Marital.Status)
levels(Marital.Status) <- list(Married=c("Male", "Maried","Married","Divorced", "widower"), 
                       Single=c("singal", "single", "sINGLE","Single", "Unmarried"))
table(Marital.Status)
Master1 <- data.frame(Master,Gender,Marital.Status)
Master2 <- Master1[c(-7,-8)]
## now from Primary dataset
##primary$Marital.Status[primary$Marital.Status == 0] <- 'Married'
##primary$Marital.Status[primary$Marital.Status == 'Divorcee'] <- 'Married'
##primary$Marital.Status[primary$Marital.Status == 'Divorced'] <- 'Married'
##primary$Marital.Status[primary$Marital.Status == 'Unmarried'] <- 'Single'

primary$Marital.Status[is.na(primary$Marital.Status)] <- 'Married'
Marital_Status<- primary$Marital.Status
Marital_Status <- factor(Marital_Status)
levels(Marital_Status) <- list(Married=c('0', "Divorcee","Divorced","Married"), 
                               Single=c("Unmarried", "single"))
primary1<- data.frame(primary,Marital_Status)

##Finding out IDs which are missmatched
##Sal_data[Sal_data$EMPLOYEE_CODE %in% p$Empcode]
##Modified_ID <- ifelse(Master$Empcode == Sal_data$EMPLOYEE_CODE,1,0)

##merge(Sal_data,Master, by = Sal_data$EMPLOYEE_CODE,  all.x = TRUE)

dih_col <- ifelse(Sal_data$EMPLOYEE_CODE %in% Master2$Empcode, Sal_data$EMPLOYEE_CODE, 0)
summary(dih_col)
View(dih_col)
str(dih_col)
dih_col1<- which(Sal_data$EMPLOYEE_CODE %in% Master2$Empcode, Sal_data$EMPLOYEE_CODE, 0)
View(dih_col1)

##Sal_integrate <-data.frame(Sal_data,Modified_ID)
sal2<-data.frame(Sal_data,dih_col)
sal2$dih_col[sal2$dih_col == 0 ] <- NA
table(is.na(sal2))
sal_final <- na.omit(sal2)
summary(sal_final)

##Integrating these two datasets

Final <- data.frame(sal_final,Master2)
Final_1 <- Final[c(-4,-5)]

##Integrating ID of three dataset.
which(unique(primary[c("Employee.ID")]))

dih_col2 <- ifelse(Sal_data$EMPLOYEE_CODE %in% Master2$Empcode,Sal_data$EMPLOYEE_CODE,
                   ifelse(Sal_data$EMPLOYEE_CODE %in% primary1$Employee.ID,Sal_data$EMPLOYEE_CODE,0))
View(dih_col2)

##sal2<-data.frame(Sal_data,dih_col2)
sal3 <- data.frame(sal2,dih_col2)
sal3$dih_col2[sal3$dih_col2 == 0 ] <- NA
table(is.na(sal3))

## Feature engineering
Gender1 <- ifelse(sal3$dih_col2 == Master2$Empcode, Master2$Gender,
                 ifelse(sal3$dih_col2 == primary1$Employee.ID, primary1$Gender, NA))

show(Gender1)
table(is.na(Gender1))

Marital_Status <- ifelse(sal3$dih_col2 == Master2$Empcode, Master2$Marital.Status,
                         ifelse(sal3$dih_col2 == primary1$Employee.ID, primary1$Marital.Status, NA))

summary(Marital_Status)

show(Marital_Status)
table(is.na(Marital_Status))

##getting the position of NA value

X <- data.frame(sal3,Marital_Status)



Match <- ifelse(sal3$EMPLOYEE_CODE %in% primary1$Employee.ID,1,0)
table(Match)


## Integrating Unit name with Sal data
Region <- Master2$Unit_Name
table(Region)
as.character(Region)

Region1 <- ifelse(sal3$EMPLOYEE_CODE %in% Master2$Empcode, as.character(Master2$Unit_Name), NA)
## Region2 <- ifelse(primary1$Employee.ID %in% Master2$Empcode, as.character(Master2$Unit_Name), NA)
table(Region1)                        

Sal4 <- data.frame(sal3,Region1)
Region2 <- ifelse(Sal4$EMPLOYEE_CODE %in% primary1$Employee.ID,as.character(Master2$Unit_Name),NA)

table(Region2)
Region_clean <- na.omit(Region2)
Data_test <- as.tibble(fread('C:/Users/7000320/Desktop/Allsec Attrition rate/test_solor.csv', sep = "\t", nrows= 8000))
write.csv(Data_test, "Data_test.csv")




s1 <- merge(Old_database,Salary, by= "Employee.ID")
s2 <- merge(s1,Test_7, by = "Employee.ID")
p1 <- read.csv(file.choose(),stringsAsFactors = FALSE, header = TRUE)

table(as.factor(p1$A))
str(p1$A)
summary(p1$ï..1000)
colnames(p1)[1] <- "Employee.ID"

s3 <- merge(Test_7,p1, by = "Employee.ID")
s4 <- merge(s3,Salary,by = "Employee.ID")

table(Test_Final_1$Resigned.y)


TRAIn_1 <- subset(TRAIN,select = -c(5,7,8))
tree_fit_1 <- rpart(Resigned.y ~., data = train_nn, method = "class",
                   control = list(minsplit = 10, 
                                  minbucket = 2))
rpart.plot(tree_fit_1)
fancyRpartPlot(tree_fit_1)



Data_python <- as.tibble(fread('C:/Users/7000320/Desktop/Allsec Attrition rate/test_solor.csv', sep = "\t", nrows= 8000))
write.csv(Dataframe1, "Allsec_python.csv")


############################################################################################

library(plotly)

set.seed(1234)
dt <- Dataframe1

p <- ggplot(dt, aes(x=Gender, y= Marital.Status)) + geom_boxplot()

p <- ggplotly(p)

p


p11 <- plot_ly(dt, x = ~ Gross_pay, color = ~ Marital.Status, type = "box")
p11

p12 <- plot_ly(dt, x = dt$Gross_pay , y = dt$Overall.Experience , color= as.numeric(dt$Overall.Experience)>10,
               type = "histogram")
p12


p13 <- plot_ly(dt, x = ~ Gross_pay, color = ~ as_date(Year_aa), type = "box")
p13

p14 <- plot_ly(dt, x = ~ Gross_pay, color = ~ Resigned.y, type = "box")
p14


## p15 <- plot_ly(dt, x = ~ Gross_pay, color = ~ Resigned.y, type = "box")


p15 <- plot_ly(dt, labels = dt$Supporting.market, values = dt$Gross_pay, type = 'pie') %>%
  layout(title = 'Market wise Gross salary distribution',
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
p15

p16 <- plot_ly(dt, labels = dt$Department.Technology, values = dt$Gross_pay, type = 'pie') %>%
  layout(title = 'Department wise Gross salary distribution',
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
p16

p17 <- plot_ly(dt, labels = dt$Highest.Educational.Qualification, values = dt$Gross_pay, type = 'pie') %>%
  layout(title = 'Education wise Gross salary distribution',
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))
p17


# Create a shareable link to your chart
## Set up API credentials: https://plot.ly/r/getting-started
Sys.setenv("plotly_username"="Pijushwd007")
Sys.setenv("plotly_api_key"="2jL5rcDU4IUsgB6ofe9a")
chart_link = plotly_POST(p17, filename="geom_boxplot/basic")
chart_link


ggplot(time_series, aes(y= Gross_pay, x=Final_Age )) +
  geom_point()


mtcars$manuf <- sapply(strsplit(rownames(mtcars), " "), "[[", 1)

p18 <- dt %>%
  group_by(dt$Sal_bin) %>%
  summarize(count = n()) %>%
  plot_ly(labels = dt$Sal_bin, values = dt$Gross_pay) %>%
  add_pie(hole = 0.6) %>%
  layout(title = "Donut charts using Plotly",  showlegend = F,
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))


p18

p19 <- dt %>%
  group_by (dt$Sal_bin) %>%
  summarize(count = n()) %>%
  plot_ly(labels = dt$Sal_bin, values = as.numeric(dt$Overall.Experience)) %>%
  add_pie(hole = 0.6) %>%
  layout(title = "Donut charts Sal bin vs Overall exp",  showlegend = F,
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))


p19

p20 <- dt %>%
  group_by (dt$Resigned.y) %>%
  summarize(count = n()) %>%
  plot_ly(labels = dt$Sal_bin, values = as.numeric(dt$Overall.Experience)) %>%
  add_pie(hole = 0.6) %>%
  layout(title = "Donut charts Sal bin vs Overall exp",  showlegend = F,
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))






Resigned_map <- c( A = 1, I = 2)
values1 <- Resigned_map[as.character(Dataframe1$Resigned.y)]

myts1 <- ts(values1, start=c(2015, 1), end=c(2017, 12), frequency=12)
myts1
plot(myts1)

fit1 <- stl(myts1, s.window="period")
plot(fit1)

monthplot(myts1)
library(forecast)
seasonplot(myts1)


##*************************************************************************************
origin1 <- "2015-01-01"
duration2 <- as.POSIXct (sprintf("2015-01-01 %s", Dataframe1$Year_aa)) - as.POSIXct("2015-01-01")
time1 <- as.POSIXct( c(0, cumsum(as.numeric(duration2))), origin=origin1)

ts1 <- xts( c(values1, tail(values1,1)),  time1)

plot(ts1, type = "s")


Resigned_number <- c(1471, 2639, 2982, 3022, 2036, 2956, 3020)
date <- as.Date(c('2015-1-1','2015-2-1','2015-3-1','2015-4-1','2015-5-1','2015-6-1','2015-7-1'))

Time_series1 <- data.frame(date,Resigned_number)

myts <- ts(Time_series1$Resigned_number, start=c(2015, 1), end=c(2015, 7), frequency=12)
myts
plot(myts)


fit <- stl(myts, t.window= 7, s.window="per", robust = TRUE)
plot(fit)

monthplot(myts)
library(forecast)
seasonplot(myts)
accuracy(fit)


fit1 <- HoltWinters(myts, beta=FALSE, gamma=FALSE)
accuracy(fit1)

library(forecast)
forecast(fit, 3)
plot(forecast(fit, 3))




###############################################################################################

TEST
TEST$Gender = as.factor(TEST$Gender)
TEST$Marital.Status = as.factor (TEST$Marital.Status)
TEST$Highest.Educational.Qualification = as.factor(TEST$Highest.Educational.Qualification)
TEST$Supporting.market= as.factor(TEST$Supporting.market)
TEST$Job.title= as.factor(TEST$Job.title)
TEST$Department.Technology = as.factor(TEST$Department.Technology)
TEST$Region_1 = as.factor(TEST$Region_1)

TRAIN$Gender = as.factor(TRAIN$Gender)
TRAIN$Marital.Status = as.factor (TRAIN$Marital.Status)
TRAIN$Highest.Educational.Qualification = as.factor(TRAIN$Highest.Educational.Qualification)
TRAIN$Supporting.market= as.factor(TRAIN$Supporting.market)
TRAIN$Job.title= as.factor(TRAIN$Job.title)
TRAIN$Department.Technology = as.factor(TRAIN$Department.Technology)
TRAIN$Region_1 = as.factor(TRAIN$Region_1)

control = trainControl(method="repeatedcv", number=10, repeats=3)
model_xyz = train(Resigned.y ~., data=TRAIN, method="rpart", preProcess="scale", trControl=control)
summary(model_xyz)

rpart.plot(model_xyz)
tree_fit_xyz <- rpart(Resigned.y ~.
                  , data = TRAIN, method = "class", na.action = na.pass,
                  control = rpart.control(minbucket = 7, cp = 0.0358))
rpart.plot(tree_fit_xyz)
fancyRpartPlot(tree_fit_xyz)

prediction1_xyz <- predict(tree_fit_xyz,TEST, type = "class")
confusionMatrix(TEST$Resigned.y,prediction1_xyz)

























































































modified_salary <- read.csv(file.choose())
colnames(modified_salary)[1] <- "Employee.ID"

Dataframe1 <- merge(Dataframe_11,modified_salary, by= "Employee.ID")
table(is.na(Dataframe1$Gross_pay))

##Data_python <- as.tibble(fread('C:/Users/7000320/Desktop/Allsec Attrition rate/test_solor.csv', sep = "\t", nrows= 8000))
write.csv(Dataframe1, "Allsec_python2.csv")

Dataframe1
Gross_pay1 <- ifelse(Dataframe1$Anual_salary > 1200000, NA , Dataframe1$Anual_salary)
table(is.na(Gross_pay1))

Dataframe_2 = data.frame(Dataframe1,Gross_pay1)
table(is.na(Dataframe_2))

Dataframe1$Sal_bin <- cut(Dataframe1$Anual_salary, breaks = c(0, seq(200000.0, 3000000.0, by = 200000.0)), labels = 0:14)
table(is.na(Dataframe1$Sal_bin))



qplot(Final_Age, data=Dataframe_2, geom="density", fill= Sal_bin, alpha=I(.5), 
      main="Desity Curve Age vs Salary", xlab="Age", 
      ylab="Density")


write.csv(Dataframe_2, "Allsec_python1.csv")

Region_1 <- ifelse(Dataframe1$Employee.ID %in% Old_database$Employee.ID, as.character(Old_database$Unit_Name),
                   ifelse(Dataframe1$Employee.ID %in% modified_salary$Employee.ID,as.character(Old_database$Unit_Name), NA))
table(is.na((Region_1)))
table(Test_Final$Region_1)
Dataframe1 <- data.frame(Dataframe1,Region_1)
Dataframe1$Region_1[Dataframe1$Region_1 == "Manila"] <- 'Chennai'
Dataframe1$Region_1[Dataframe1$Region_1 == "Dallas"] <- 'Chennai'

Test_Final_modified <- subset(Dataframe1,select = -c(1,4,12,14,15,16,19,20))
Test_Final_modified$Resigned.y <-  ifelse(Test_Final_modified$Resigned.y == 'A',1,0)
table(Test_Final_modified$Resigned.y)
table(is.na(Test_Final_modified))
Data1 <- sort(sample(nrow(Test_Final_modified), nrow(Test_Final_modified)* .7))
TRAIN_modified <- Test_Final_1[Data1,]
TEST_modified <- Test_Final_1[-Data1,]


Test_nn_m <- Test_Final_modified
Test_nn_m$Gender <- as.numeric(as.factor(Test_nn_m$Gender))
Test_nn_m$Marital.Status <-as.numeric(as.factor(Test_nn_m$Marital.Status))
Test_nn_m$Highest.Educational.Qualification<- as.numeric(as.factor(Test_nn_m$Highest.Educational.Qualification))
#Test_nn_m$Overall.Experience<-as.numeric(as.factor(Test_nn_m$Overall.Experience))
Test_nn_m$Department.Technology <- as.numeric(as.factor(Test_nn_m$Department.Technology))
#Test_nn_m$Length.of.Service <- as.numeric(as.factor(Test_nn_m$Length.of.Service))
Test_nn_m$Job.title <- as.numeric(as.factor(Test_nn_m$Job.title))
Test_nn_m$Supporting.market <- as.numeric(as.factor(Test_nn_m$Supporting.market))
#Test_nn_m$Sal_bin <- as.numeric(as.factor(Test_nn_m$Sal_bin))
Test_nn_m$Region_1 <- as.numeric(as.factor(Test_nn_m$Region_1))

Data12 <- sort(sample(nrow(Test_nn_m), nrow(Test_nn_m)* .7))
train_nn_m <- Test_nn_m[Data12,]
test1_nn_m <- Test_nn_m[-Data12,]


library(rpart)
library(rpart.plot)
library(rattle)
## Cross-Validation
##Gender = as.numeric(as.factor(Test))

control = trainControl(method="repeatedcv", number=10, repeats=3)
model = train(Resigned.y ~., data=train_nn_m, method="rpart", preProcess="scale", trControl=control)
summary(model)

tree_fit <- rpart(Resigned.y ~.,
                  data = train_nn_m, method = "class", na.action = na.pass,
                  control = rpart.control(minbucket = 4, cp = 0.018))
rpart.plot(tree_fit)
fancyRpartPlot(tree_fit)

prediction1 <- predict(tree_fit,test1_nn_m, type = "class")
prediction1
confusionMatrix(test1_nn_m$Resigned.y,prediction1)
confusionMatrix()

tree_fit11 <- rpart(Resigned.y ~ Supporting.market + Job.title + Highest.Educational.Qualification + Overall.Experience + Length.of.Service +
                      Final_Age + Sal_bin + Region_1, data = train_nn_m, method = "class",
                   control = list(minsplit = 20, 
                                  minbucket = 7, cp = 0.008, maxcompete = 4, maxsurrogate = 5, 
                                  usesurrogate = 2, surrogatestyle = 0, maxdepth = 30, xval = 0))

rpart.plot(tree_fit11)
fancyRpartPlot(tree_fit11)
prediction_m1 = predict(tree_fit11,test1_nn_m,type = "class")
prediction_modified <- predict(tree_fit11,test1_nn_m, type = "class")
confusionMatrix(test1_nn_m$Resigned.y,prediction_m1)
table(prediction_modified)

## Logistic regression

Reg_fit <- glm(Resigned.y ~., family = binomial(link = "logit"), data = train_nn_m, control = list(maxit = 50))
summary(Reg_fit)

Regression_fit <- glm(Resigned.y ~ Gender + Highest.Educational.Qualification + Overall.Experience + Length.of.Service +
                        Sal_bin + Region_1 , family = binomial(link = "logit"),
                      data = train_nn_m)
summary(Regression_fit)
anova(Regression_fit, test = "Chisq")

Reg_predict <- predict(Regression_fit, newdata = subset(test1_nn_m,select = c(1,3,4,6,11,12)), type = 'response')

Reg_predict <- ifelse(Reg_predict > 0.5,1,0)

library(ROCR)
library(Metrics)
library(caTools)
library(rattle)
confusionMatrix(test1_nn_m$Resigned.y,Reg_predict)
perf <- performance(Reg_predict, measure = "tpr", x.measure = "fpr")
plot(perf)
auc(test1_nn_m$Resigned.y,Reg_predict)

##Neural Network

library(neuralnet)
set.seed(2)
NN = neuralnet(Resigned.y ~ Gender + Marital.Status + Highest.Educational.Qualification + Overall.Experience +
                 Length.of.Service +Sal_bin + Region_1, train_nn_m, hidden = 2,
               linear.output = TRUE )
NN
plot(NN)


NN1 <- nnet(Resigned.y ~ Gender + Marital.Status + Highest.Educational.Qualification + Overall.Experience +
              Length.of.Service +Sal_bin + Region_1, data = train_nn_m, size = 3, rang = 0.1, decay = 0.1, maxit = 20)

Predict2 <- predict(NN1,test1_nn_m, type = "raw")
Predict2 <- ifelse(Predict2 > 0.5,1,0)
confusionMatrix(Predict2, test1_nn_m$Resigned.y)


###Random Forest

Random_forest_fit <- randomForest(Resigned.y ~ Gender + Marital.Status + Highest.Educational.Qualification + Overall.Experience +
                                    Length.of.Service +Sal_bin + Region_1, data = train_nn_m, ntree= 200,
                                  importance = FALSE, na.action = na.omit)


print(Random_forest_fit)

Random_forest_fit1 <- randomForest(Resigned.y ~., data = train_nn_m, ntree= 500,
                                   importance = TRUE, mtry= 11, na.action = na.omit)
print(Random_forest_fit1)

plot(Random_forest_fit1)
importance(Random_forest_fit1, type = 1)
varImpPlot(Random_forest_fit1)

Pred_rf <- predict(Random_forest_fit1, type = "response")
Pred_rf <- ifelse(Pred_rf > 0.3,1,0)
confusionMatrix(Pred_rf, test1_nn_m$Resigned.y)
str(test1_nn$Resigned.y)

## SVM

library(e1071)

## Crossvalidation
tuned = tune.svm(Resigned.y ~., data = train_nn_m, gamma = 10^-2, cost = 10^2, tunecontrol=tune.control(cross=5))
summary(tuned)
obj <- tune(svm, Resigned.y~., data = train_nn_m, 
            ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)),
            tunecontrol = tune.control(sampling = "fix"))
summary(obj)

svm_fit <- svm(Resigned.y ~., data = train_nn_m, kernel= "radial", cost = 4, gamma = .05 )
summary(svm_fit)
pred_svm <- predict(svm_fit, data= test1_nn_m)
table(pred_svm1)
pred_svm2 <- predict(svm_fit, data= train_nn_m)
pred_svm2 <- ifelse(pred_svm2 > 0.5,1,0)
confusionMatrix(train_nn_m$Resigned.y,pred_svm2)
confusionMatrix(test1_nn_m$Resigned.y,pred_svm)
points(train_nn_m$Resigned.y, pred_svm2, col= 'blue', pch = 4)

##pred_svm2_f <- ifelse(pred_svm2> 0.5,1,0)
##confusionMatrix(test1_nn$Resigned.y,pred_svm2_f)
##tune.control(random = FALSE, nrepeat = 2, repeat.aggregate = mean,
##             sampling = c("cross", "fix", "bootstrap"), sampling.aggregate = mean,
##             sampling.dispersion = sd,
##             cross = 10, fix = 2/3, nboot = 10, boot.size = 9/10, best.model = TRUE,
##             performances = TRUE, error.fun = NULL)

tune.svm(x, y = NULL, data = NULL, degree = NULL, gamma = NULL, coef0 = NULL,
         cost = NULL, nu = NULL, class.weights = NULL, epsilon = NULL, ...)
best.svm(x, tunecontrol = tune.control(), ...)






























