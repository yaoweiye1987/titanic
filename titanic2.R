# load libraries
library(caret)
library(pROC)

#################################################
# data prep
#################################################

indat<-read.csv("/Users/weiyeyao/Documents/avant-analytics-interview/TitanicData.csv")
indat2<-read.csv("/Users/weiyeyao/Documents/avant-analytics-interview/TitanicData.csv")

titanicDF <- rbind(indat, indat2)
# load data
#titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')
titanicDF$Title <- ifelse(grepl('Mr',titanicDF$Name),'Mr',ifelse(grepl('Mrs',titanicDF$Name),'Mrs',ifelse(grepl('Miss',titanicDF$Name),'Miss','Nothing'))) 
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)

# miso format
#titanicDF <- titanicDF[c('PClass', 'Age',    'Sex',   'Title', 'Survived')]
titanicDF <- titanicDF[c('PassengerId', 'Pclass', 'Title','Sex','Age','Fare','Embarked', 'Survived')]

# dummy variables for factors/characters
titanicDF$Title <- as.factor(titanicDF$Title)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))
print(names(titanicDF))


# what is the proportion of your outcome variable?
prop.table(table(titanicDF$Survived))

# save the outcome for the glmnet model
tempOutcome <- titanicDF$Survived  

# generalize outcome and predictor variables
outcomeName <- 'Survived'
predictorsNames <- names(titanicDF)[names(titanicDF) != outcomeName]

names(getModelInfo())

titanicDF$Survived <- ifelse(titanicDF$Survived==1,'yes','no')



################################################
# glmnet model
################################################

# pick model gbm and find out what type of model it is
getModelInfo()$glmnet$type

# save the outcome for the glmnet model
titanicDF$Survived  <- tempOutcome

# split data into training and testing chunks
set.seed(1234)
splitIndex <- createDataPartition(titanicDF[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]

# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=3, returnResamp='none')

# run model
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='glmnet',  metric = "RMSE", trControl=objControl)

# get predictions on your testing data
predictions <- predict(object=objModel, testDF[,predictorsNames])

library(pROC)
auc <- roc(testDF[,outcomeName], predictions)
print(auc$auc)

postResample(pred=predictions, obs=testDF[,outcomeName])

# find out variable importance
summary(objModel)
plot(varImp(objModel,scale=F))

# find out model details
objModel

# display variable importance on a +/- scale 
vimp <- varImp(objModel, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  


