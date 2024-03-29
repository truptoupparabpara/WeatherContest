

### 읽어오기

library(tidyverse)
setwd("C:/temp")
bev_all<- read.csv("bev_all_weather.csv")
bev_coco<- read.csv("bev_coco_weather.csv")
bev_ia<- read.csv("bev_ia_weather.csv")
bev_latte<- read.csv("bev_latte_weather.csv")
bev_juice<- read.csv("bev_juice_weather.csv")
bev_icetea<- read.csv("bev_icetea_weather.csv")
head(bev_all)
bev_all<-bev_all[-c(1, 8)]
bev_coco<-bev_coco[-c(1, 4)]
bev_latte<-bev_latte[-c(1, 4)]
bev_icetea<-bev_icetea[-c(1, 4)]
bev_juice<-bev_juice[-c(1,4)]
bev_ia<-bev_ia[-c(1,4)]


### 산점도


pairs(bev_all[-1])



### EDA 및 기초분석 



## 날씨

# 비
ggplot(data=bev_all)+geom_point(aes(datetime, weather.sum_rn))
bev_all[bev_all["weather.sum_rn"]>50,1]
## 비 많이 온 날은 대부분 여름
## 강수량은 대부분 0에 가까운듯

#기온
ggplot(data=bev_all)+geom_point(aes(datetime, weather.avg_ta))
ggplot(data=bev_all)+geom_line(aes(weather.avg_ta, juice), colour="#330066")+labs(x="temperature")+ggtitle("temperature vs. juice")
ggplot(data=bev_all)+geom_line(aes(weather.avg_ta, ia), colour="#993300")+labs(x="temperature", y="ice americano")+ggtitle("temperature vs. iceamericano")
ggplot(data=bev_all)+geom_line(aes(weather.avg_ta, latte), colour="#CC6600")+labs(x="temperature")+ggtitle("temperature vs. latte")
ggplot(data=bev_all)+geom_line(aes(weather.avg_ta, coco), colour="#663300")+labs(x="temperature")+ggtitle("temperature vs. cocoa")
ggplot(data=bev_all)+geom_line(aes(weather.avg_ta, icetea), colour="#663300", alpha=0.5)+geom_smooth(aes(weather.avg_ta, icetea))+labs(x="temperature")+ggtitle("temperature vs. icetea")


# 습도
ggplot(data=bev_all)+geom_point(aes(datetime, weather.avg_rhm))
ggplot(data=bev_all)+geom_line(aes(weather.avg_rhm, juice), colour="#330066")+labs(x="humidity")+ggtitle("humidity vs. juice")
ggplot(data=bev_all)+geom_line(aes(weather.avg_rhm, ia), colour="#993300")+labs(x="humidity", y="ice americano")+ggtitle("humidity vs. ice americano")
ggplot(data=bev_all)+geom_line(aes(weather.avg_rhm, latte), colour="#CC6600")+labs(x="humidity")+ggtitle("humidity vs. latte")
ggplot(data=bev_all)+geom_line(aes(weather.avg_rhm, coco), colour="#663300")+labs(x="humidity")+ggtitle("humidity vs. cocoa")
ggplot(data=bev_all)+geom_line(aes(weather.avg_rhm, icetea), colour="#663300", alpha=0.5)+geom_smooth(aes(weather.avg_rhm, icetea),method="lm")+labs(x="humidity")+ggtitle("humidity vs. icetea")


# 바람
ggplot(data=bev_all)+geom_point(aes(datetime, weather.avg_wa))
ggplot(data=bev_all)+geom_line(aes(weather.avg_wa, juice), colour="#330066")+labs(x="wind speed")+ggtitle("wind speed vs. juice")
ggplot(data=bev_all)+geom_line(aes(weather.avg_wa, ia), colour="#993300")+labs(x="wind speed", y="ice americano")+ggtitle("wind speed vs. ice americano")
ggplot(data=bev_all)+geom_line(aes(weather.avg_wa, latte), colour="#CC6600")+labs(x="wind speed")+ggtitle("wind speed vs. latte")
ggplot(data=bev_all)+geom_line(aes(weather.avg_wa, coco), colour="#663300")+labs(x="wind speed")+ggtitle("wind speed vs. cocoa")
ggplot(data=bev_all)+geom_line(aes(weather.avg_wa, icetea), colour="#663300", alpha=0.5)+geom_smooth(aes(weather.avg_wa, icetea), method="lm")+labs(x="wind speed")+ggtitle("wind speed vs. ice tea")








##### 기온과 유관한것으로 보임 




##### 7일, 365일 단위로 주기성 보임


  
  
  
  ### 이상치 처리
  
  
  
  
# 검색량이 주위 날보다 급격하게 높은 하루는 이상치로 간주




# ia
ggplot(data=bev_ia)+geom_point(aes(datetime, ia))

#coco
ggplot(data=bev_coco)+geom_point(aes(datetime, coco))
idx<-which(diff(bev_coco$coco)>50)+1
bev_coco[idx,]$coco=(bev_coco[idx-1,]$coco+bev_coco[idx+1,]$coco)/2

#juice
ggplot(data=bev_juice)+geom_point(aes(datetime, juice))
idx<-which(diff(bev_juice$juice)>50)+1
bev_juice[idx,]$juice=(bev_juice[idx-1,]$juice+bev_juice[idx+1,]$juice)/2

# latte
ggplot(data=bev_latte)+geom_point(aes(datetime, latte))
idx<-which(diff(bev_latte$latte)>50)+1
bev_latte[idx,]$latte=(bev_latte[idx-1,]$latte+bev_latte[idx+1,]$latte)/2

# icetea
ggplot(data=bev_icetea)+geom_point(aes(datetime, icetea))


  
  
  ### training / test set 

training_ind<-floor(nrow(bev_all)*0.8)
# training set
ia_train<-bev_ia[1:training_ind,]
coco_train<-bev_coco[1:training_ind,]
latte_train<-bev_latte[1:training_ind,]
juice_train<-bev_juice[1:training_ind,]
icetea_train<-bev_icetea[1:training_ind,]

# test set
ia_test<-bev_ia[(training_ind+1):nrow(bev_all),]
coco_test<-bev_coco[(training_ind+1):nrow(bev_all),]
latte_test<-bev_latte[(training_ind+1):nrow(bev_all),]
juice_test<-bev_juice[(training_ind+1):nrow(bev_all),]
icetea_test<-bev_icetea[(training_ind+1):nrow(bev_all),]

  
  ### 단순 회귀
  
  
  
  
  # 아이스아메리카노

lm_ia<-lm(ia~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=bev_ia)
best_lm_ia<-step(lm_ia)
summary(best_lm_ia)
plot(best_lm_ia$residuals)
ggplot(data=bev_ia)+geom_point(aes(datetime, ia))+geom_point(aes(datetime, predict(best_lm_ia)), colour="red", alpha=0.5)
```


#코코아류

lm_coco<-lm(coco~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=bev_coco)
best_lm_coco<-step(lm_coco)
summary(best_lm_coco)
plot(best_lm_coco$residuals)
ggplot(data=bev_coco)+geom_point(aes(datetime, coco))+geom_point(aes(datetime, predict(best_lm_coco)), colour="red", alpha=0.5)
```


# 아이스티

lm_icetea<-lm(icetea~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=bev_icetea)
best_lm_icetea<-step(lm_icetea)
summary(best_lm_icetea)
plot(best_lm_icetea$residuals)
ggplot(data=bev_icetea)+geom_point(aes(datetime, icetea))+geom_point(aes(datetime, predict(best_lm_icetea)), colour="red", alpha=0.5)


# 주스류

lm_juice<-lm(juice~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=bev_juice)
best_lm_juice<-step(lm_juice)
summary(best_lm_juice)
plot(best_lm_juice$residuals)
ggplot(data=bev_juice)+geom_point(aes(datetime, juice))+geom_point(aes(datetime, predict(best_lm_juice)), colour="red", alpha=0.5)


# 라떼류


lm_latte<-lm(latte~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=bev_latte)
best_lm_latte<-step(lm_latte)
summary(best_lm_latte)
plot(best_lm_latte$residuals)
ggplot(data=bev_latte)+geom_point(aes(datetime, latte))+geom_point(aes(datetime, predict(best_lm_latte)), colour="red", alpha=0.5)



### linear regression / training & test 



# 아이스아메리카노

lm_ia<-lm(ia~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=ia_train)
best_lm_ia<-step(lm_ia)
summary(best_lm_ia)
lm_ia_fore<-predict(best_lm_ia, newdata=ia_test)
mse_ia<-sqrt((sum((ia_test$ia-lm_ia_fore)^2))/nrow(ia_test))
mse_ia
mae_ia=sum(abs(ia_test$ia-lm_ia_fore))/nrow(ia_test)
mae_ia
ggplot(data=ia_test)+geom_point(aes(datetime, ia, colour="black"))+geom_point(aes(datetime, lm_ia_fore, colour="red"), alpha=0.5)+scale_color_discrete(name = "color", labels = c("real", "predict"))+theme(legend.position=c(0.9, 0.8))



# 코코아류

lm_coco<-lm(coco~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=coco_train)
best_lm_coco<-step(lm_coco)
summary(best_lm_coco)
lm_coco_fore<-predict(best_lm_coco, newdata=coco_test)
mse_coco<-sqrt(sum((coco_test$coco-lm_coco_fore)^2)/nrow(coco_test))
mse_coco
mae_coco=sum(abs(coco_test$coco-lm_coco_fore))/nrow(coco_test)
mae_coco
ggplot(data=coco_test)+geom_point(aes(datetime, coco, colour="black"))+geom_point(aes(datetime, lm_coco_fore, colour="red"), alpha=0.5)+scale_color_discrete(name = "color", labels = c("real", "predict"))+theme(legend.position=c(0.9, 0.8))


#아이스티

lm_icetea<-lm(icetea~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=icetea_train)
best_lm_icetea<-step(lm_icetea)
summary(best_lm_icetea)
lm_icetea_fore<-predict(best_lm_icetea, newdata=icetea_test)
mse_icetea<-sqrt(sum((icetea_test$icetea-lm_icetea_fore)^2)/nrow(icetea_test))
mse_icetea
mae_icetea=sum(abs(icetea_test$icetea-lm_icetea_fore))/nrow(icetea_test)
mae_icetea
ggplot(data=icetea_test)+geom_point(aes(datetime, icetea, colour="black"))+geom_point(aes(datetime, lm_icetea_fore, colour="red"), alpha=0.5)+scale_color_discrete(name = "color", labels = c("real", "predict"))+theme(legend.position=c(0.9, 0.8))


# 주스류

lm_juice<-lm(juice~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=juice_train)
best_lm_juice<-step(lm_juice)
summary(best_lm_juice)
lm_juice_fore<-predict(best_lm_juice, newdata=juice_test)
mse_juice<-sqrt(sum((juice_test$juice-lm_juice_fore)^2)/nrow(juice_test))
mse_juice
mae_juice=sum(abs(juice_test$juice-lm_juice_fore))/nrow(juice_test)
mae_juice
ggplot(data=juice_test)+geom_point(aes(datetime, juice, colour="black"))+geom_point(aes(datetime, lm_juice_fore, colour="red"), alpha=0.5)+scale_color_discrete(name = "color", labels = c("real", "predict"))+theme(legend.position=c(0.9, 0.8))


# 라떼류

lm_latte<-lm(latte~weather.max_ta+weather.avg_rhm+rn_label+weather.avg_wa+weekday+as.character(season), data=latte_train)
best_lm_latte<-step(lm_latte)
summary(best_lm_latte)
lm_latte_fore<-predict(best_lm_latte, newdata=latte_test)
mse_latte<-sqrt(sum((latte_test$latte-lm_latte_fore)^2)/nrow(latte_test))
mse_latte
mae_latte=sum(abs(latte_test$latte-lm_latte_fore))/nrow(latte_test)
mae_latte
ggplot(data=latte_test)+geom_point(aes(datetime, latte, colour="black"))+geom_point(aes(datetime, lm_latte_fore, colour="red"), alpha=0.5)+scale_color_discrete(name = "color", labels = c("real", "predict"))+theme(legend.position=c(0.9, 0.8))


  
  ### ADL모형 fitting (lagged regression)
  
  
  
  ##### dynlm 사용 
  
  

library(dynlm)



# 아이스아메리카노

dyn_ia<-dynlm(ia~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(ia,1), data=bev_ia)
dyn_ia<-step(dyn_ia)
summary(dyn_ia)
dyn_ia_fit<-fitted(dyn_ia, data=bev_ia)
dyn_ia_mse<-sqrt(sum((bev_ia[1:(nrow(bev_ia)-1),]$ia-dyn_ia_fit)^2)/nrow(bev_ia))
dyn_ia_mse
dyn_ia_mae<-sum(abs(bev_ia[1:(nrow(bev_ia)-1),]$ia-dyn_ia_fit))/nrow(bev_ia)
dyn_ia_mae
ggplot(data=bev_ia[2:(nrow(bev_ia)),])+geom_point(aes(datetime, ia, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_ia_fit, colour="predict"),alpha=0.5)


# 코코아류

dyn_coco<-dynlm(coco~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(coco,1), data=bev_coco)
dyn_coco<-step(dyn_coco)
summary(dyn_coco)
dyn_coco_fit<-fitted(dyn_coco, data=bev_coco)
dyn_coco_mse<-sqrt(sum((bev_coco[1:(nrow(bev_coco)-1),]$coco-dyn_coco_fit)^2)/nrow(bev_coco))
dyn_coco_mse
dyn_coco_mae<-sum(abs(bev_coco[1:(nrow(bev_coco)-1),]$coco-dyn_coco_fit))/nrow(bev_coco)
dyn_coco_mae
ggplot(data=bev_coco[2:(nrow(bev_coco)),])+geom_point(aes(datetime, coco, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_coco_fit, colour="predict"),alpha=0.5)



# 아이스티

dyn_icetea<-dynlm(icetea~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(icetea,1), data=bev_icetea)
dyn_icetea<-step(dyn_icetea)
summary(dyn_icetea)
dyn_icetea_fit<-fitted(dyn_icetea, data=bev_icetea)
dyn_icetea_mse<-sqrt(sum((bev_icetea[1:(nrow(bev_icetea)-1),]$icetea-dyn_icetea_fit)^2)/nrow(bev_icetea))
dyn_icetea_mae<-sum(abs(bev_icetea[1:(nrow(bev_icetea)-1),]$icetea-dyn_icetea_fit))/nrow(bev_icetea)
dyn_icetea_mae
ggplot(data=bev_icetea[2:(nrow(bev_icetea)),])+geom_point(aes(datetime, icetea, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_icetea_fit, colour="predict"),alpha=0.5)




# 주스류

dyn_juice<-dynlm(juice~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(juice,1), data=bev_juice)
dyn_juice<-step(dyn_juice)
summary(dyn_juice)
dyn_juice_fit<-fitted(dyn_juice, data=bev_juice)
dyn_juice_mse<-sqrt(sum((bev_juice[1:(nrow(bev_juice)-1),]$juice-dyn_juice_fit)^2)/nrow(bev_juice))
dyn_juice_mse
dyn_juice_mae<-sum(abs(bev_juice[1:(nrow(bev_juice)-1),]$juice-dyn_juice_fit))/nrow(bev_juice)
dyn_juice_mae
ggplot(data=bev_juice[2:(nrow(bev_juice)),])+geom_point(aes(datetime, juice, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_juice_fit, colour="predict"),alpha=0.5)



# 라떼류

dyn_latte<-dynlm(latte~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(latte,1), data=bev_latte)
dyn_latte<-step(dyn_latte)
summary(dyn_latte)
dyn_latte_fit<-fitted(dyn_latte, data=bev_latte)
dyn_latte_mse<-sqrt(sum((bev_latte[1:(nrow(bev_latte)-1),]$ia-dyn_latte_fit)^2)/nrow(bev_latte))
dyn_latte_mae<-sum(abs(bev_latte[1:(nrow(bev_latte)-1),]$ia-dyn_latte_fit))/nrow(bev_latte)
dyn_latte_mae
ggplot(data=bev_latte[2:(nrow(bev_latte)),])+geom_point(aes(datetime, latte, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_latte_fit, colour="predict"),alpha=0.5)




### training / test 



# 아이스아메리카노

dyn_ia_t<-dynlm(ia~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(ia,1), data=ia_train)
dyn_ia_t<-step(dyn_ia_t)
summary(dyn_ia_t)
dyn_ia_t_fit<-predict(dyn_ia_t, newdata=ia_test)
dyn_ia_mse<-sqrt(sum((ia_test[2:(nrow(ia_test)),]$ia-dyn_ia_t_fit[-1])^2)/(nrow(ia_test)-1))
dyn_ia_mse
dyn_ia_mae<-sum(abs(ia_test[2:(nrow(ia_test)),]$ia-dyn_ia_t_fit[-1]))/(nrow(ia_test)-1)
dyn_ia_mae
ggplot(data=ia_test[2:(nrow(ia_test)),])+geom_point(aes(datetime, ia, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_ia_t_fit[-1], colour="predict"),alpha=0.5)



# 코코아류

dyn_coco_t<-dynlm(coco~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(coco,1), data=coco_train)
dyn_coco_t<-step(dyn_coco_t)
summary(dyn_coco_t)
dyn_coco_t_fit<-predict(dyn_coco_t, newdata=coco_test)
dyn_coco_mse<-sqrt(sum((coco_test[2:(nrow(coco_test)),]$coco-dyn_coco_t_fit[-1])^2)/(nrow(coco_test)-1))
dyn_coco_mse
dyn_coco_mae<-sum(abs(coco_test[2:(nrow(coco_test)),]$coco-dyn_coco_t_fit[-1]))/(nrow(coco_test)-1)
dyn_coco_mae
ggplot(data=coco_test[2:(nrow(coco_test)),])+geom_point(aes(datetime, coco, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_coco_t_fit[-1], colour="predict"),alpha=0.5)


# 아이스티

dyn_icetea_t<-dynlm(icetea~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(icetea,1), data=icetea_train)
dyn_icetea_t<-step(dyn_icetea_t)
summary(dyn_icetea_t)
dyn_icetea_t_fit<-predict(dyn_icetea_t, newdata=icetea_test)
dyn_icetea_mse<-sqrt(sum((icetea_test[2:(nrow(icetea_test)),]$icetea-dyn_icetea_t_fit[-1])^2)/(nrow(icetea_test)-1))
dyn_icetea_mae<-sum(abs(icetea_test[2:(nrow(icetea_test)),]$icetea-dyn_icetea_t_fit[-1]))/(nrow(icetea_test)-1)
dyn_icetea_mae
ggplot(data=icetea_test[2:(nrow(icetea_test)),])+geom_point(aes(datetime, icetea, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_icetea_t_fit[-1], colour="predict"),alpha=0.5)


# 주스류

dyn_juice_t<-dynlm(juice~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(juice,1), data=juice_train)
summary(dyn_juice_t)
dyn_juice_t<-step(dyn_juice_t)
dyn_juice_t_fit<-predict(dyn_juice_t, newdata=juice_test)
dyn_juice_mse<-sqrt(sum((juice_test[2:(nrow(juice_test)),]$juice-dyn_juice_t_fit[-1])^2)/(nrow(juice_test)-1))
dyn_juice_mse
dyn_juice_mae<-sum(abs(juice_test[2:(nrow(juice_test)),]$juice-dyn_juice_t_fit[-1]))/(nrow(juice_test)-1)
dyn_juice_mae
ggplot(data=juice_test[2:(nrow(juice_test)),])+geom_point(aes(datetime, juice, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_juice_t_fit[-1], colour="predict"),alpha=0.5)


# 라떼류

dyn_latte_t<-dynlm(latte~weather.max_ta+weather.avg_wa+weather.avg_rhm+rn_label+weekday+as.character(season)+lag(latte,1), data=latte_train)
dyn_latte_t<-step(dyn_latte_t)
summary(dyn_latte_t)
dyn_latte_t_fit<-predict(dyn_latte_t, newdata=latte_test)
dyn_latte_mse<-sqrt(sum((latte_test[2:(nrow(latte_test)),]$latte-dyn_latte_t_fit[-1])^2)/(nrow(latte_test)-1))
dyn_latte_mse
dyn_latte_mae<-sum(abs(latte_test[2:(nrow(latte_test)),]$latte-dyn_latte_t_fit[-1]))/(nrow(latte_test)-1)
dyn_latte_mae
ggplot(data=latte_test[2:(nrow(latte_test)),])+geom_point(aes(datetime, latte, colour="real"), alpha=0.3)+geom_point(aes(datetime, dyn_latte_t_fit[-1], colour="predict"),alpha=0.5)

