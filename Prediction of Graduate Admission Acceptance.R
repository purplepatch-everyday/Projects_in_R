# 대학원 입시 데이터를 활용한 합격류 예측 

data <- read.csv("university.csv", header = TRUE)
head(data,3)
str(data)

# 결측치 확인 
sum(is.na(data))

# 유니크 값 확인 
# 대략적인 값의 분포 확인 가능
unique(data$GRE.Score)
unique(data$TOEFL.Score)
unique(data$University.Rating)
unique(data$SOP)
unique(data$LOR)
unique(data$Research)

# 제일 중요한것은 taget값 확인
unique(data$Chance.of.Admit)
max(data$Chance.of.Admit)
min(data$Chance.of.Admit)

table(data$University.Rating)
table(data$Research)

# 변수 산점도 
plot(data)

# 회귀분석을 통해 합격률 예측 

set.seed(2021)

newdata <- data
View(newdata)
train_ratio <- 0.8
datatotal <- sort(sample(nrow(newdata),nrow(newdata)*train_ratio))
train <- newdata[datatotal,]
test <- newdata[-datatotal,]

# 로지스틱 회귀 분석

library(caret)
ctrl <- trainControl(method = "repeatedcv",repeats = 5)
logistic <- train(Chance.of.Admit~.,
            data = train,
            method = "glm",
            trControl=ctrl,
            preProcess=c("center","scale"),
            metric="RMSE")
# glm = generalized linear model
# 데이터 전처리 필요! 
# 평가 방식: RMSE
# 분류 문제일땐, accuracy
# 예측 문제일땐, RMSE

logistic

logistic_pred <- predict(logistic,newdata=test)
logistic_pred

# 엘라스틱넷
ctrl <- trainControl(method = "repeatedcv",repeats = 5)
elastic <- train(Chance.of.Admit~.,
                  data = train,
                  method = "glmnet",
                  trControl=ctrl,
                  preProcess=c("center","scale"),
                  metric="RMSE")
elastic
# 평가하기
# 최적화 값
# alpha 가 0 일때 L1, Lasso Regression 
#          1 일때 L2, Ridge Regression
# lambda 전체 제약식의 크기 

elasticnet_pred <- predict(elastic, newdata=test)
postResample(pred=elasticnet_pred,obs = test$Chance.of.Admit)

# RMSE가 낮을수록 좋음 

# 랜덤포레스트 

rf <- train(Chance.of.Admit~.,
                 data = train,
                 method = "rf",
                 trControl=ctrl,
                 preProcess=c("center","scale"),
                 metric="RMSE")
rf
plot(rf)

# 평가하기
rf_pred <- predict(rf, newdata=test)
postResample(pred=rf_pred,obs = test$Chance.of.Admit)

# 선형 서포트 벡터 머신 

svmlinear <- train(Chance.of.Admit~.,
            data = train,
            method = "svmLinear",
            trControl=ctrl,
            preProcess=c("center","scale"),
            metric="RMSE")
svmlinear

# 예측하기 
svmlinear_pred <- predict(svmlinear,newdata=test)
postResample(pred = svmlinear_pred, obs = test$Chance.of.Admit)

#커널 서포트 벡터 머신

svmkernel <- train(Chance.of.Admit~.,
                   data = train,
                   method = "svmPoly",
                   trControl=ctrl,
                   preProcess=c("center","scale"),
                   metric="RMSE")
svmkernel
# degree = 차수
# scale =  내적크기 조절
# C = 비용

plot(svmkernel)

# 예측하기 
svmkernel_pred <- predict(svmkernel, newdata = test)
postResample(pred = svmkernel_pred,obs = test$Chance.of.Admit)

# 모형결과 비교 하면 최종적으로 엘라스틱넷이 가장 좋은 모형으로 평가가 된다. 
