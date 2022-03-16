rm(list=ls())
library(quantmod)
library(TTR)

# Gathering data ----------------------------------------------------------


set.seed(555)
#s&p500 dataset
spy <- getSymbols("^GSPC", src = "yahoo",
                  from = '2010-01-01', to = '2021-01-01', auto.assign = FALSE)
#Checking how the dataset looks
head(spy)


# Defining 10 features ----------------------------------------------------


#V1: Spy Volume
volume <- spy[,5]


#V2: HML(Highest price minus lowest price)
hml <- spy[,2]-spy[,3]


#V3: EMA(Exponential moving average)
ema <- EMA(spy[,1], n=20)


#V4: RSI(Relative Strength Index)
rsi <- RSI(spy[,1], n=14)


#V5: SMA(Simple moving average)
sma <- SMA(spy[,1], n=20)


#V6: MACD
macd <- MACD(spy[,1], fast = 12, nFast = 12, nSlow=26, nSig=9)


#V8: Log Return
lr <- ROC(spy[,1], n=12)


#V9: Average Directional Index
adx <- ADX(HLC(spy), n=3)


#V10: Average True Range
atr <- ATR(HLC(spy), n=10)


# Defining Target Variable ------------------------------------------------


#Closing price - opening price (Is today a positive or negative day)
price_change<-spy[,4]-spy[,1]
price_change<-as.numeric(price_change)

#Making sure results are as expected
head(price_change)

#1 is UP 0 is DOWN
#Outcome variable
target<- ifelse(price_change <= 0, "1", "0")

#Making sure results are as expected
head(target)


# Dataset Building --------------------------------------------------------


#Dataset of the target(Up or down day) and technical indicators
df <- data.frame(volume, hml, ema, rsi, sma, macd, lr, adx, atr)

#Remove na value rows from dataset and create dataset
df <- data.frame(df, target)
df <- df[-c(1:35),]
head(df)

# Decision Tree Building ---------------------------------------------------


library("rpart")
library("caret")
library("janitor")


#Model Fitting
trControl <- trainControl(method="repeatedcv",number=10,repeats=10)

#Parameters
grid <- expand.grid(.cp=0.01)


#Fitting model
df <- df[,-1]
cart.fit <- train(target~., data=df, trControl=trControl,
                  method="rpart",
                  tuneGrid=grid)


# Evaluating Model --------------------------------------------------------


cart.fit
cart.fit$finalModel

#Most important variables
cart.fit$finalModel$variable.importance

df.2 <- df[,c("rsi","DIn", "target")]
head(df.2)

#Small model with 2 variables
grid <- expand.grid(.cp=0.1)

cart.fit.2 <- train(target~., data=df.2, trControl=trControl,
                    method="rpart",
                    tuneGrid=grid)

cart.fit.2$finalModel


# Decision Plot ------------------------------------------------------------


# Define the function:
decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  # p <- predict(model, g, type = predict_type)
  p <- predict(model,newdata=g)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  invisible(z)
}

decisionplot(cart.fit.2,data=df.2,
             class="target",
             main="CART Decision Boundary")


# Overfitting -------------------------------------------------------------


#Small number to overfit
grid <- expand.grid(.cp=0.000001)

cart.fit.2 <- train(target~., data=df.2, trControl=trControl,
                    method="rpart",
                    tuneGrid=grid,
                    control=rpart.control(minsplit = 2))

cart.fit.2$finalModel
decisionplot(cart.fit.2,data=df.2,
             class="target",
             main="CART Decision Boundary")


# Hyperparamter -----------------------------------------------------------


#Parameters
trControl <- trainControl(method="repeatedcv",number=10,repeats=10)

grid <- expand.grid(.cp=10^seq(-8,10,by=0.5))
dim(grid)

#What grid looks like
head(grid)

#Fitting model
df <- df[,-1]
cart.fit <- train(target~., data=df, trControl=trControl,
                  method="rpart",
                  tuneGrid=grid, metric="Accuracy")

names(cart.fit)
cart.fit$finalModel

#Accuracy
my.cp <- log(cart.fit$results$cp)
my.Accuracy <- cart.fit$results$Accuracy
plot(x=my.cp,y=my.Accuracy)

#Optimal Alpha
cp.opt <- cart.fit$results$cp[which.max(cart.fit$results$Accuracy)]
message(sprintf("Optimal alpha is %6.5f.",cp.opt))

#Tuned closer to optimal alpha:

trControl <- trainControl(method="repeatedcv",number=10,repeats=10)

# Set the tuning parameters
grid <- expand.grid(.cp=seq(0,0.004,length=100))

cart.fit.tight <- train(target~., data=df,
                        trControl=trControl,
                        method="rpart",
                        tuneGrid=grid,
                        metric="Accuracy")

#Chart of results
my.cp.tight <- (cart.fit.tight$results$cp)
my.Accuracy.tight <- cart.fit.tight$results$Accuracy
plot(x=my.cp.tight,
     y=my.Accuracy.tight)

#Optimal cp
cp.opt.tight <- cart.fit.tight$results$cp[which.max(cart.fit.tight$results$Accuracy)]
message(sprintf("Optimal alpha is %6.5f.",cp.opt.tight))

print(cart.fit$finalModel$variable.importance)
