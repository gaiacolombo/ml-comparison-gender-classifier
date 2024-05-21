#installa librerie se non sono già installate
packages <- c("rstudioapi", "ggplot2", "lattice", "caret", "rpart", "rattle",
              "neuralnet", "e1071", "C50", "ROCR", "pROCR", "naivebayes")
for (i in 1:length(packages)) {
  if(packages[i] %in% rownames(installed.packages()) == FALSE) {
    install.packages(packages[i])
  }
}

#importa librerie
library(rstudioapi)
library(ggplot2)
library(lattice)
library(caret)
library(rpart)
library(rattle)
library(neuralnet)
library(e1071)
library(C50)
library(ROCR)
library(pROC)
library(naivebayes)


#carica file csv
path = dirname(getSourceEditorContext()$path)
dataset_path = paste(path,"/gender_classification_v7.csv", sep="")
dataset = read.csv(dataset_path, sep = ",")


#trasforma gender in una variabile categorica
dataset$gender = factor(dataset$gender)


###2: ANALISI DELLE COVARIATE###


##2.1: UNIVARIATE##

#2.1.1: INDICI DI STATISTICA DESCRITTIVA#

#ricava gli indici di statistica descrittiva
summary(dataset)

#crea un dataset x con tutte le covariate, ma senza il target
x = dataset[,1:7]

#crea un vettore y con solo i valori del target
y = dataset[,8]

#2.1.2: BOXPLOT#

#crea i boxplot di tutto il dataset
par(mfrow = c(1,7)) #divide lo schermo in più pannelli
for(i in 1:7) {boxplot(x[,i], main=names(dataset)[i])} #crea un boxplot per ogni covariata
par(mfrow = c(1,1))

#2.1.3: DENSITY PLOT#

#density plot
featurePlot(x, y, plot="density", scales=list(x=list(relation="free"), y=list(relation="free")), auto.key=list(columns=3), col=c("#de7ed3","#7edec9"))


##2.2: MULTIVARIATE##

#rende tutte le variabili booleane da numeriche a fattoriali
dataset$long_hair = factor(dataset$long_hair)
dataset$nose_wide = factor(dataset$nose_wide)
dataset$nose_long = factor(dataset$nose_long)
dataset$lips_thin = factor(dataset$lips_thin)
dataset$distance_nose_to_lip_long = factor(dataset$distance_nose_to_lip_long)
x = dataset[,1:7]

#plot multivariate
featurePlot(x=x, y=y, plot="pairs", auto.key=list(columns=3), col=c("#de7ed3","#7edec9"))



###3: MODELLI DI MACHINE LEARNING###

#funzione per dividere gli attributi del dataset in training e test
split.data = function(data, p = 0.7, s = 1){
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1] * p)], ]
  test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
  return(list(train=train, test=test)) }

#richiama la funzione split.data
datalist = split.data(dataset)

#crea il database trainset e il database testset
trainset = datalist$train
testset = datalist$test

#mostra quanti maschi e quante femmine ci sono in trainset
table(trainset$gender)
prop.table(table(trainset$gender))

#trasforma la covariata prediction nel testset in 0 per i maschi e 1 per le femmine, in modo da poter confrontare il target con prediction
testset$gender <- as.factor(ifelse(testset$gender=="Male", 0, 1))


##3.1: BASELINE MODEL##

#aggiunge la covariata prediction nel testset, e per ogni istanza mette casualmente 0 (maschio) o 1 (femmina)
testset$prediction = rep(sample(c(0,1), replace=TRUE, size=1500))
testset$prediction = factor(testset$prediction)

#crea la confusion matrix della previsione a valori casuali
confusion.matrix = table(testset$gender, testset$prediction)
confusion.matrix

#calcola l'accuratezza del modello
sum(diag(confusion.matrix))/sum(confusion.matrix)

#crea una tabella che mostra la relazione tra distance_nose_to_lip_long e il genere
prop.table(table(trainset$distance_nose_to_lip_long, trainset$gender), 1)

#mette prediction maschio se distance_nose_to_lip_long è 1, femmina altrimenti
testset$prediction <- ifelse(testset$distance_nose_to_lip_long==1, 0, 1)
testset$prediction = factor(testset$prediction)

#calcola l'accuratezza del modello
confusion.matrix = table(testset$gender, testset$prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)


##3.2: ALBERI DI DECISIONE##

#3.2.1: ALBERO DI DECISIONE CON 6 FOGLIE#

#crea e mostra il decision tree basandosi sulle classe
decisionTree = rpart(gender ~ ., data=trainset, method="class")
fancyRpartPlot(decisionTree)

#mette in prediction il risultato dell'albero (0 per maschio, 1 per femmina)
testset$prediction <- predict(decisionTree, testset, type = "class")
testset$prediction <- as.factor(ifelse(testset$prediction=="Male", 0, 1))

#calcola l'accuratezza dell'albero a 6 foglie
confusion.matrix = table(testset$gender, testset$prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

#stampa il complexity parameter e ne traccia il grafico
printcp(decisionTree)
plotcp(decisionTree)

#3.2.2: ALBERO DI DECISIONE CON 2 FOGLIE#

#taglia l'albero dopo solamente un nodo e lo mostra
prunedDecisionTree = prune(decisionTree, cp=.16)
fancyRpartPlot(prunedDecisionTree)

#calcola l'accuratezza dell'albero a 2 foglie
testset$prediction <- predict(prunedDecisionTree, testset, type = "class")
testset$prediction <- as.factor(ifelse(testset$prediction=="Male", 0, 1))
confusion.matrix = table(testset$gender, testset$prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

#3.2.3: ALBERO DI DECISIONE CON INFORMATION GAIN#

#albero basato su IG, uso solo 3 variabili, lo testo, e ottengo comunque un ottimo risultato
decisionTreeIG = rpart(gender ~ distance_nose_to_lip_long + nose_long + lips_thin, data=trainset, method="class", parms=list(split='information'))
fancyRpartPlot(decisionTreeIG)
testset$prediction <- predict(decisionTreeIG, testset, type = "class")
testset$prediction <- as.factor(ifelse(testset$prediction=="Male", 0, 1))
confusion.matrix = table(testset$gender, testset$prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)


##3.3:RETI NEURALI##

#rendo numeriche le colonne fattoriali del trainset
trainset$long_hair = as.numeric(trainset$long_hair)
trainset$long_hair = trainset$long_hair - 1
trainset$nose_wide = as.numeric(trainset$nose_wide)
trainset$nose_wide = trainset$nose_wide - 1
trainset$nose_long = as.numeric(trainset$nose_long)
trainset$nose_long = trainset$nose_long - 1
trainset$lips_thin = as.numeric(trainset$lips_thin)
trainset$lips_thin = trainset$lips_thin - 1
trainset$distance_nose_to_lip_long = as.numeric(trainset$distance_nose_to_lip_long)
trainset$distance_nose_to_lip_long = trainset$distance_nose_to_lip_long - 1

#aggiunge la colonna male e female
trainset$male = trainset$gender == "Male"
trainset$female = trainset$gender == "Female"

#fa la rete con 4 neuroni hidden e la mostra
network = neuralnet(male + female~ long_hair + forehead_width_cm + forehead_height_cm + nose_wide + nose_long + lips_thin + distance_nose_to_lip_long, trainset, hidden=4)
plot(network)

#crea una tabella con le probabilità in output (percentuale di essere male o female)
testset$long_hair = as.numeric(testset$long_hair)
testset$long_hair = testset$long_hair - 1
testset$nose_wide = as.numeric(testset$nose_wide)
testset$nose_wide = testset$nose_wide - 1
testset$nose_long = as.numeric(testset$nose_long)
testset$nose_long = testset$nose_long - 1
testset$lips_thin = as.numeric(testset$lips_thin)
testset$lips_thin = testset$lips_thin - 1
testset$distance_nose_to_lip_long = as.numeric(testset$distance_nose_to_lip_long)
testset$distance_nose_to_lip_long = testset$distance_nose_to_lip_long - 1
net.predict = compute(network, testset[-8])$net.result

#calcola quale percetuale delle due è più alta e setta l'attributo di conseguenza
net.prediction = c(0, 1)[apply(net.predict, 1, which.max)]

#crea la tabella tipo quella di confusione e ne calcola accuracy
predict.table = table(testset$gender, net.prediction)
sum(diag(predict.table))/sum(predict.table)
predict.table

#crea i plot dei pesi generalizzati per i maschi
par(mfrow=c(2, 4))
gwplot(network,selected.covariate="long_hair")
gwplot(network,selected.covariate="forehead_width_cm")
gwplot(network,selected.covariate="forehead_height_cm")
gwplot(network,selected.covariate="nose_wide")
gwplot(network,selected.covariate="nose_long")
gwplot(network,selected.covariate="lips_thin")
gwplot(network,selected.covariate="distance_nose_to_lip_long")


##3.4:NAIVE BAYES##

#toglie colonne numeriche e rende fattoriale gli attributi di trainset e testset
trainset = trainset[, c(1,4,5,6,7,8)]
trainset$long_hair = factor(trainset$long_hair)
trainset$nose_wide = factor(trainset$nose_wide)
trainset$nose_long = factor(trainset$nose_long)
trainset$lips_thin = factor(trainset$lips_thin)
trainset$distance_nose_to_lip_long = factor(trainset$distance_nose_to_lip_long)
testset$long_hair = factor(testset$long_hair)
testset$nose_wide = factor(testset$nose_wide)
testset$nose_long = factor(testset$nose_long)
testset$lips_thin = factor(testset$lips_thin)
testset$distance_nose_to_lip_long = factor(testset$distance_nose_to_lip_long)

#esegue l'algoritmo e lo applica a testset
classifier = naiveBayes(trainset[, 1:5], trainset[, 6])
testset$prediction = predict(classifier, testset)

#fattorizza la variabile prediction e crea la matrice di confusione
testset$prediction <- as.factor(ifelse(testset$prediction=="Male", 0, 1))
confusion.matrix = table(testset$gender, testset$prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)



###4:COMPARAZIONE DEI MODELLI###


##4.1: ROC##

#facciamo train con albero, rete e bayas
control = trainControl(method = "repeatedcv", number = 10,repeats = 3,
                       classProbs = TRUE, summaryFunction = twoClassSummary)

#albero
rpart.model= train(gender ~ ., data = trainset, method = "rpart", metric = "ROC",
                   trControl = control)
rpart.probs = predict(rpart.model, testset[,! names(testset) %in% c("prediction")],
                      type = "prob")
rpart.ROC = roc(response = testset[,c("gender")], predictor =rpart.probs$Female,
                levels = levels(testset[,c("gender")]))

rpart.prob = predict(rpart.model, testset[,! names(testset) %in% c("gender")])
rpart.prob <- as.factor(ifelse(rpart.prob=="Male", 0, 1))
result_rpart = confusionMatrix(rpart.prob, testset[,c("gender")], mode = "prec_recall")
result_rpart

#rete neurale con nnet
rneural.model= train(gender ~ ., data = trainset, method = "nnet", metric = "ROC",
                     trControl = control)
rneural.probs = predict(rneural.model, testset[,! names(testset) %in% c("prediction")],
                        type = "prob")
rneural.ROC = roc(response = testset[,c("gender")], predictor =rneural.probs$Female,
                  levels = levels(testset[,c("gender")]))

rneural.prob = predict(rneural.model, testset[,! names(testset) %in% c("gender")])
rneural.prob <- as.factor(ifelse(rneural.prob=="Male", 0, 1))
result_rneural = confusionMatrix(rneural.prob, testset[,c("gender")], mode = "prec_recall")
result_rneural

#bayes
rbayes.model= train(gender ~ ., data = trainset, method = "naive_bayes", metric = "ROC",
                    trControl = control)
rbayes.probs = predict(rbayes.model, testset[,! names(testset) %in% c("prediction")],
                       type = "prob")
rbayes.ROC = roc(response = testset[,c("gender")], predictor =rbayes.probs$Female,
                 levels = levels(testset[,c("gender")]))

rbayes.prob = predict(rbayes.model, testset[,! names(testset) %in% c("gender")])
rbayes.prob <- as.factor(ifelse(rbayes.prob=="Male", 0, 1))
result_rbayes = confusionMatrix(rbayes.prob, testset[,c("gender")], mode = "prec_recall")
result_rbayes

#grafico curve ROC
plot(rpart.ROC,  col="blue")
rpart.ROC
plot(rneural.ROC, add=TRUE, col="green")
rpart.ROC
plot(rbayes.ROC, add=TRUE, col="red")
rbayes.ROC

#confronto intervalli di confidenza
cv.values = resamples(list(rpart = rpart.model, rneural = rneural.model, rbayes = rbayes.model))
summary(cv.values)
dotplot(cv.values, metric = "ROC")
bwplot(cv.values, layout = c(3, 1))
splom(cv.values,metric="ROC")


##4.2: MISURE DI PERFORMANCE##

par(mfrow = c(1,1))
testset$gender <- as.factor(ifelse(testset$gender==0, "Male", "Female"))
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]],
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

#albero
tree.model= train(gender ~ ., data = trainset, method = "rpart")
tree.pred = predict(tree.model, testset[,! names(testset) %in% c("gender")])
table(tree.pred, testset[,c("gender")])
treeresult = confusionMatrix(tree.pred, testset[,c("gender")])

treefit=rpart(gender~ ., data=trainset, method="class")
pred=predict(treefit,testset[, !names(testset) %in% c("gender")], probability=TRUE)
pred.to.roc = pred[, 2]

pred.rocr = prediction(pred.to.roc, testset$gender)

acctree.perf = performance(pred.rocr, measure = "acc")
ind = which.max( slot(acctree.perf, "y.values")[[1]] )
acc = slot(acctree.perf, "y.values")[[1]][ind]
cutoff = slot(acctree.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))

#rete
nn.model= train(gender ~ ., data = trainset, method = "nnet")
nn.pred = predict(nn.model, testset[,! names(testset) %in% c("gender")])
table(nn.pred, testset[,c("gender")])
nnresult = confusionMatrix(nn.pred, testset[,c("gender")])

pred.to.roc = net.predict[, 1]

pred.rocr = prediction(pred.to.roc, testset$gender)

accnn.perf = performance(pred.rocr, measure = "acc")
plot(accnn.perf)
ind = which.max( slot(accnn.perf, "y.values")[[1]] )
acc = slot(accnn.perf, "y.values")[[1]][ind]
cutoff = slot(accnn.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))

#bayes
bayes.model= train(gender ~ ., data = trainset, method = "naive_bayes")
bayes.pred = predict(bayes.model, testset[,! names(testset) %in% c("gender")])
table(bayes.pred, testset[,c("gender")])
bayesresult = confusionMatrix(bayes.pred, testset[,c("gender")])

pred=predict(bayes.model, testset[,! names(testset) %in% c("gender")],
             type = "prob")
pred.to.roc = pred[, 2]

pred.rocr = prediction(pred.to.roc, testset$gender)

accbayes.perf = performance(pred.rocr, measure = "acc")
ind = which.max( slot(accbayes.perf, "y.values")[[1]] )
acc = slot(accbayes.perf, "y.values")[[1]][ind]
cutoff = slot(accbayes.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))


##4.3: MAXIMUM ACCURACY##

plot(accnn.perf, col="green")
plot(acctree.perf, add=TRUE,  col="blue")
plot(accbayes.perf, add=TRUE, col="red")


##4.2: TEMPI DI TRAINING##

#tabella con i tempi di calcolo della 10-fold
cv.values$timings
