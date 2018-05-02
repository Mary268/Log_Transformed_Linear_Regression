#Author: Menglu Wang
#
#719 Case 7: Demand Model Estimation
#Multiple Linear Regression
library("data.table")
library(rpart)
library(randomForest)
blink_data = read.csv("/Users/marywang/Documents/MSCI_719/719_case7/demand_variables.csv")
blink_testdata = read.csv("/Users/marywang/Documents/MSCI_719/719_case7/demand_testdata.csv")
#shift function: back 1 unit as lag1
blink_variables = data.frame(sales_units_lag1  = blink_data$sales_units,
                                 discount_per      = blink_data$discount_per,
                                 discount_per_lag1 = blink_data$discount_per_lag1,
                                 promo_week        = blink_data$promo_week_flg,
                                 age               = blink_data$age)
blink_response = blink_data$response

blink_testvariables = data.frame(sales_units_lag1  = blink_testdata$sales_units,
                                 discount_per      = blink_testdata$discount_per,
                                 discount_per_lag1 = blink_testdata$discount_per_lag1,
                                 promo_week        = blink_testdata$promo_week_flg,
                                 age               = blink_testdata$age)
blink_testresponse = blink_testdata$response

x1 = (blink_variables$sales_units_lag1) 
x2 = (blink_variables$discount_per)
x3 = (blink_variables$discount_per_lag1) 
x4 = blink_variables$promo_week 
x5 = (blink_variables$age)

x6 = (blink_variables$sales_units_lag1)*blink_variables$promo_week 
x7 = (blink_variables$discount_per)*blink_variables$promo_week 
x8 = (blink_variables$discount_per_lag1)*blink_variables$promo_week 
x9 = (blink_variables$age)*blink_variables$promo_week 

x10 = log(x1)
x11 = log(x5)

y1 = (blink_testvariables$sales_units_lag1) 
y2 = (blink_testvariables$discount_per)
y3 = (blink_testvariables$discount_per_lag1) 
y4 = blink_testvariables$promo_week 
y5 = (blink_testvariables$age)

y6 = (blink_testvariables$sales_units_lag1)*blink_testvariables$promo_week 
y7 = (blink_testvariables$discount_per)*blink_testvariables$promo_week 
y8 = (blink_testvariables$discount_per_lag1)*blink_testvariables$promo_week 
y9 = (blink_testvariables$age)*blink_testvariables$promo_week 

y10 = log(y1)
y11 = log(y5)


lm = lm((blink_response) ~  x1+x2+x3+x4+x5)

lm_full = lm((blink_response) ~  x1+x2+x3+x4+x5+x1*x4+x2*x4+x3*x4+x5*x4)

lm_log_age = lm(log(blink_response) ~  x10+x2+x3+x4+x11)

lm_log = lm(log(blink_response) ~ log(blink_variables$sales_units_lag1) + blink_variables$discount_per
        + blink_variables$discount_per_lag1 + blink_variables$promo_week + blink_variables$age)

lm_x = lm(log(blink_response) ~ log(x1)+x2+x3+x4+x5)

lm_x.pred = exp(predict(lm_x, newdata = data.frame(data.frame(x1  = y10,
                                                              x2 = y2,
                                                              x3 = y3,
                                                              x4 = y4,
                                                              x5 = y5))))

lm_x.error = lm_x.pred - blink_testresponse

ts = ts(blink_response, frequency = 12)
ts.arima = arima(x = blink_response, order = c(2,1,4))
ts.hw = hw(ts, h = 19, seasonal = "multiplicative", initial = "optimal")
#Forecast of holt's Winter time series Ajustment is:
ts.hw.pred = c(45.39126,
19.80756,
17.30723,
20.58619,
20.20059,
21.92133,
24.04279,
28.78956,
34.36,
38.07363,
51.97118,
55.92343,
52.82392,
23.00734,
20.06599,
23.82461,
23.33726,
25.28173,
27.68194)

lm_x.pred_seaadj = lm_x.pred + 2*ts.hw.pred

lm.pred = predict(lm, newdata = data.frame(data.frame(x1  = y1,
                                                          x2 = y2,
                                                          x3 = y3,
                                                          x4 = y4,
                                                          x5 =y5)))
lm_full.pred = predict(lm_full, newdata =  data.frame(data.frame(x1  = y1,
                                                                 x2 = y2,
                                                                 x3 = y3,
                                                                 x4 = y4,
                                                                 x5 = y5,
                                                                 x6 = y6,
                                                                 x7 = y7,
                                                                 x8 = y8,
                                                                 x9 = y9))) 
lm_log_age.pred = exp(predict(lm_log_age, newdata = data.frame(data.frame(x10 = y10,
                                                                      x2 = y2,
                                                                      x3 = y3,
                                                                      x4 = y4,
                                                                      x11 = y11))))

#ts.arima.forecast = forecast(ts.arima, h = 19)
pairs(~blink_data$sales_units+blink_data$discount_per+blink_data$discount_per_lag1+blink_data$promo_week_flg+blink_data$age,data=blink_data, 
             main="BLINK Training Scatterplot Matrix",cex = 0.5,col = "chocolate4")

