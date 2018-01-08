
library(caret)
library(pROC)
indat<-read.csv("/Users/weiyeyao/Documents/avant-analytics-interview/TitanicData.csv")
indat2<-read.csv("/Users/weiyeyao/Documents/avant-analytics-interview/TitanicData.csv")
total <- rbind(indat, indat2)
total$Sex <- as.factor(total$Sex)
total$Embarked <- as.factor(total$Embarked)
total$Title <- ifelse(grepl('Mrs',total$Name),'Mrs',ifelse(grepl('Mr', total$Name),'Mr',ifelse(grepl('Miss',total$Name),'Miss','Nothing'))) 
total$Age[is.na(total$Age)] <- median(total$Age, na.rm=T)
total$Title <- as.factor(total$Title)
total <- total[c('PassengerId', 'Pclass', 'Title','Sex','Age','Fare','Embarked', 'Survived')]

dummy <- dummyVars("~.", data = total, fullRank = F)
str(dummy)
total <- as.data.frame(predict(dummy,total))


prop.table(table(total$Survived))
tempOutcome <- total$Survived
outcomeName <- 'Survived'
predictorNames <- names(total)[names(total) != outcomeName]

getModelInfo()$glmnet$type

set.seed(34560)
splitIndex <- createDataPartition (total[,outcomeName], p =0.70, list = FALSE, times = 1)
trainTitanic  <- total[splitIndex,]
testTitanic   <- total[-splitIndex,]
#objControl <-trainControl (method = 'cv', number = 10, returnResamp = 'none')
objModel <- train(trainTitanic[,predictorNames], trainTitanic[,outcomeName], method='glmnet',  metric = "RMSE")

predictions <- predict (object = objModel, testTitanic[, predictorNames])
auc <- roc(testTitanic[,outcomeName], predictions)
print(auc$auc)

summary(objModel)
objModel
plot(varImp(objModel,scale=F))

