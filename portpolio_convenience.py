#!/usr/bin/env python
# coding: utf-8

# In[1]:


#데이터 전처리에 필요한 모듈 불러오기

import pandas as pd
import re


# In[2]:


convenience=pd.read_csv(r'C:\Users\hiyoe\Downloads\행정동별_편의점분야_소비인구_201127.csv')#편의점 정보 불러오기


convenience.head() #편의점 정보 헤드5개만

convenience.info() #편의점 정보  확인

convenience.isnull().sum() #결측치 확인  signgu_nm 컬럼에서만 나타난다.

convenience['bntr_nm'][convenience['signgu_nm'].isnull()].unique() #결측치가 있는 도시는 세종특별자치시,우리는 서울시만 

convenience = convenience[convenience['bntr_nm']=='서울특별시'].reset_index(drop=True) #서울만 사용하기,인덱스 리셋

#20~40대 만 사용하기.
convenience=convenience[convenience['year_se'].str.contains(r'^[2-4]')]
# p = re.compile('[2-4].') #정규식을 이용해 20~40대 컴파일
# convenience=convenience[convenience['year_se'].apply(lambda x : p.match(x)!=None)] # apply lambda를 사용할 수 도 있음.

# 일자별 데이터보다 연월의 데이터가 필요하여서 convenience['de'] 를 '%Y%m' 형식의 문자로 변경
convenience['de'] = [str(i)[:6] for i in convenience['de']]

#필요한 정보 그룹별 cnsmr_popltn_co의 합을 만들자 
convenience = convenience.groupby(['gov_dn_cd','signgu_nm','adstrd_nm','de','sex_se','year_se']).sum().reset_index()

#다시 헤드 5개만
convenience.head()


# In[3]:


coffee=pd.read_csv(r'C:\Users\hiyoe\Downloads\행정동별_커피분야_소비인구_201127.csv') #커피소비인구 불러오기

icecream=pd.read_csv(r'C:\Users\hiyoe\Downloads\행정동별_제과_아이스크림분야_소비인구_201127.csv') #아이스크림 소비인구

#convenience 와 coffee와 icecream의 형식과 컬럼명 까지 똑같다. 그래서 편한 데이터 전처리를 위하여 함수로 만들기.
def data_processing(data):
    data= data[data['bntr_nm']=='서울특별시'].reset_index(drop=True) #서울시만 가져오기
    data=data[data['year_se'].str.contains(r'^[2-4]')] #20~40대 정보만 추리기
    data['de'] = [str(i)[:6] for i in data['de']] #날짜 년월로 변경
    data = data.groupby(['gov_dn_cd','signgu_nm','adstrd_nm','de','sex_se','year_se']).sum().reset_index() #필요한 정보만
    return(data)

coffee=data_processing(coffee) #커피소비인구 데이터 전처리
icecream=data_processing(icecream) #아이스크림 소비인구 데이터 전처리

#3개의 테이블을 하나로 merge시키자
#그전 컬럼이름이 완전 같아서 이름을 바꿔주자.

convenience=convenience.rename(columns={'cnsmr_popltn_co':'con_popltn_co'})
coffee=coffee.rename(columns={'cnsmr_popltn_co':'coffee_popltn_co'})
icecream=icecream.rename(columns={'cnsmr_popltn_co':'ice_popltn_co'})

#대칭 차집합을 통해 빠진 지역이 있나 확인
list(set(convenience['adstrd_nm'].unique()) ^ set(coffee['adstrd_nm'].unique()) )
list(set(convenience['adstrd_nm'].unique()) ^ set(icecream['adstrd_nm'].unique()) )

# 3개 테이블 merge
union=pd.merge(convenience,coffee,how='outer',on=['gov_dn_cd','signgu_nm','adstrd_nm','de','sex_se','year_se'])
union=pd.merge(union,icecream,how='outer',on=['gov_dn_cd','signgu_nm','adstrd_nm','de','sex_se','year_se'])

#merge후 결측치 없음
union.isnull().sum()

# 날짜 정보도 2020.08~2020.10 이므로 나누는 의미가 별로 없다.
# 기간 동안 총 누적 사용인원으로 만들어야 겠다.
union=union.drop('de',axis=1)

pd.options.display.float_format = '{:.4f}'.format #숫자는 소수점 4자리만 나타내자

union=union.groupby(['gov_dn_cd','signgu_nm','adstrd_nm','sex_se','year_se']).sum().reset_index(drop=False) #필요한 부분만 

#2020.08~2020.10는 총 92일임 , 하루 평균으로 만들어 놓자.
union['con_popltn_co']=union['con_popltn_co']/92
union['coffee_popltn_co']=union['coffee_popltn_co']/92
union['ice_popltn_co']=union['ice_popltn_co']/92

union.head()


# In[4]:


#전국 2020 연간 평균 소득정보
income=pd.read_csv(r'C:\Users\hiyoe\Downloads\가구_특성정보_(+소득정보)_211203.csv')
income=income[['adstrd_cd','ave_income_amt']] #행정동 코드가 있으므로 다른 분류가 필요 없다.

#다만 소득은 법정동 기준 평균이지만, 법정동 인구를 모르니 행정동별 표본평균으로 정리 해야한다.
income=income.groupby('adstrd_cd').mean().reset_index(drop=False) #행정동별 표본평균으로 정리

#컬럼값 union과 맞춰주기
income.columns=['gov_dn_cd','signgu_ave_income']
income.head()


# In[5]:


import numpy as np

#2020년 연령별 구별 행정동별 전체 인구수
pop=pd.read_excel(r'C:\Users\hiyoe\Downloads\Report.xls')

#종로5·6가동 가운데 온점을 .로 변경
pop['동']=pop['동'].str.replace('종로5·6가동','종로5.6가동')

#필요한 부분만 가져오자 20~40대 , 행정구역, 성별
pop = pop[['자치구','동','구분','20~24세', '25~29세','30~34세','35~39세','40~44세','45~49세']]

#컬럼명을 위 union과 같은 형태로 변경.
pop.columns=['signgu_nm','adstrd_nm','sex_se','20','25','30','35','40','45']

#행정구역별 합계 , 성별 계를 제외
pop=pop[(pop['sex_se']!='계')&(pop['adstrd_nm']!='합계')&(pop['adstrd_nm']!='소계')]

pop=pd.melt(pop,id_vars=['signgu_nm','adstrd_nm','sex_se'],var_name='year_se',value_name='pop_s')

#여자는 F, 남자는 M으로 변경.
pop['sex_se']=pop['sex_se'].apply(lambda x : 'M' if x=='남자' else 'F')

pop.head()


# In[6]:


#모든 데이터 merge
union_=pd.merge(union,pop,how='left',on=['signgu_nm','adstrd_nm','sex_se','year_se'])
union_=pd.merge(union_,income,how='left',on=['gov_dn_cd'])

union_r = union_ #복사하여 사용.

#결측치를 확인.
union_r.isnull().sum() 

#잠실 4동,6동의 평균수입이 누락되있음. 
union_r[union_['signgu_ave_income'].isnull()]

#잠실 4동6동는 잠실*동의 평균 수입으로 대체.
union_r=union_.fillna(union_r[union_r['adstrd_nm'].str.contains(r'잠실')][['adstrd_nm','signgu_ave_income']].mean())

#값 정보 확인
union_r.info()

#인구정보 float로 변경
union_r=union_r.astype({'pop_s':'float'})
union_r


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
#만들어진 자료로 편의점 이용에 대한 시각화

#성별 편의점 이용 합계 데이터프레임 만들기
sex_con_sum=union_r[['sex_se','con_popltn_co']].groupby('sex_se').sum()

sns.set_style('whitegrid') #하얀색 그리드 설정
sns.set_palette("Pastel1") #파스텔 톤 팔래트로 설정

fig = plt.figure(figsize=(15,5)) #fig생성

#제목과 그래프 배치
ax1 = fig.add_subplot(1,3,1)
ax1.set_title('Convenience store use by age')
ax2 = fig.add_subplot(1,3,2)
ax2.set_title('Convenience store use by sex')
ax3 = fig.add_subplot(1,3,3)
ax3.set_title('Convenience store use by age&sex')

#값 그래프에 넣기
sns.barplot(x='year_se',y='con_popltn_co',data=union_r,ax=ax1)
ax2.pie(sex_con_sum.con_popltn_co,labels=sex_con_sum.index,autopct='%.1f%%',explode=[0,0.1])
sns.barplot(x='year_se',y='con_popltn_co',hue='sex_se',data=union_r,ax=ax3)

plt.show()


#편의점 이용을 MAP으로 나타내기
#!pip install folium #Folium설치
import folium
import json

gov_dn_cd_sum=union_r[['gov_dn_cd','con_popltn_co']]
gov_dn_cd_sum=gov_dn_cd_sum.astype({'gov_dn_cd':'str'})
gov_dn_cd_sum=gov_dn_cd_sum.groupby('gov_dn_cd')['con_popltn_co'].sum().reset_index()
gov_dn_cd_sum.columns=['adm_cd2','con_popltn_co']

geo_json = r'https://raw.githubusercontent.com/vuski/admdongkor/master/ver20201001/HangJeongDong_ver20201001.geojson'

# params ---- 1.
center = [37.541, 126.990]
tiles = ['cartodbpositron', 'Stamen Toner', 'OpenStreetMap']

# visualization ---- 2.
m = folium.Map(
    location = [center[0], center[1]],
    zoom_start = 11,
    tiles = tiles[0])

folium.Choropleth(geo_data=geo_json,
                 name='choropleth',
                 data=gov_dn_cd_sum,
                  columns=['adm_cd2','con_popltn_co'],
                  key_on='properties.adm_cd2',
                  fill_color='Reds',
                  fill_opacity=0.5,
                  line_opacity=0.2
                 ).add_to(m)

m

#편의점은 20대가 많이 이용하며,여성보다는 남성이 많이 이용한다. 남녀간의 비율은 전 연령이 비슷한 것으로 보여진다. 
#오피스가 많은 지역에서 편의점 이용이 높아 보인다.


# In[8]:


#다른 수치들간의 관계는 어떻게 될까?
#피어슨관계로 살펴보기.
pearson_union=union_r.iloc[:,1:].corr(method='pearson')
fig_2=plt.figure(figsize=(6,5))
fig_2.add_subplot(1,1,1)
sns.heatmap(pearson_union,annot=True)



#아이스크림 판매와 강한 양의 관계를 가지고 있다. 그다음으로는 인구가 영향을 주었다. 나머지는 미미한 관계를 가짐.


# In[9]:


sns.pairplot(union_r.iloc[:,1:])
#산점도로 더욱 직접적으로 살펴 볼 수 도 있다.


# In[10]:


#아이스크림판매량과의 관계를 더욱 살펴보자
sns.set_palette("Pastel2") #색상 설정 파스텔 2

plt.figure(figsize=(12,4))

for i,j in enumerate(['con_popltn_co','ice_popltn_co']): #편의점과 아이스크림의 연령을 보기위해
    plt.subplot(1,2,i+1)
    plt.tight_layout()
    plt.title(j)
    sns.barplot(x='year_se',y=j,data=union_r)

plt.figure(figsize=(12,4))
for i,j in enumerate(['con_popltn_co','ice_popltn_co']): #편의점과 아이스크림의 연령을 그리고 성별까지 보기위해
    plt.subplot(1,2,i+1)
    plt.tight_layout()
    plt.title(j)
    sns.barplot(x='year_se',y=j,data=union_r,hue='sex_se')

    
#20~34까지는 추이가 비슷한 것을 보아, 아이스크림을 편의점에서 살 확률이 높으며,
# 35~49세까지는 다른 곳에서 구매 하는 것으로 추측할 수 있음.

# 편의점과 아이스크림 구매횟수가 가장 많은 25세를 기준으로 본다면
# 극단적으로 아이스크림을 모두 편의점에서만 구매한다 가정하여도
# 남성은 일평균 약 700회 편의점 이용 중 아이스크림 구매 횟수 약 50회. 최대 7.1% 비율
# 여성은 일평균 약 400회 편의점 이용 중 아이스크림 구매 횟수 약 90회. 최대  22.5% 비율로 

# #20~34세 여성들이 아이스크림을 구매하러 편의점에 갈 확률이 높은 편이다.


# In[11]:


#머신러닝을 이용한 예측

#필요한 모델 import
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#복사해서 사용하자.
union_m=union_r

union_m.dtypes #타입 확인.

union_m=union_m.astype({'pop_s':'float'}) #pop_s 인구를 float로 변경

#float인 값들 con_popltn_co,coffee_popltn_co ice_popltn_co signgu_ave_income 이상치 확인
iqr_list=['con_popltn_co','coffee_popltn_co', 'ice_popltn_co', 'signgu_ave_income']

#이상치를 보기위해 박스플롯
for i in iqr_list:
    plt.figure(figsize=(4,4))
    sns.boxplot(data=union_m[i])
    plt.title(i)
    plt.show()
#이상치가 많다.

#이상치 제거 함수를 만들고
def outline_iqr(data):
    q1, q3 = np.percentile(data,[25,75])
    iqr=q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    return np.where((data>upper_bound)|(data<lower_bound))

#이상치가 있는 인덱스 추출
con_iqr = outline_iqr(union_m['con_popltn_co'])[0]
coff_iqr = outline_iqr(union_m['coffee_popltn_co'])[0]
ice_iqr = outline_iqr(union_m['ice_popltn_co'])[0]
pop_iqr = outline_iqr(union_m['pop_s'])[0]
#이상치를 모으자
iqr_idx = np.concatenate([con_iqr,coff_iqr,ice_iqr,pop_iqr])

#이상치 제거
data = union_m.drop(iqr_idx,axis=0).reset_index(drop=True)
data


# In[12]:


#회귀 분석 모델링 

#회귀 분석에 사용할 컬럼을 분류한다.

#수치형 (다중공선성이 높았던 ice_popltn_co은 제외 시키고 진행한다.)
#다중 공선성이 있으면 독립변수의 공분산 행렬의 조건수(conditional number)가 증가한다.
#결과의 오버피팅으로 이어질 수 있다.
col_num = ['coffee_popltn_co', 'pop_s', 'signgu_ave_income']

#범주형 중 value_counts가 5개 이하인 것들만 사용 'gov_dn_cd','signgu_nm','adstrd_nm',는 제외
col_cat = ['sex_se','year_se']

#타겟값
col_num = ['con_popltn_co']

#범주형 자료는 원핫인코더를 사용하여 전처리 한다.
data=pd.get_dummies(data=union_m,columns=['sex_se'],prefix='sex')
data=pd.get_dummies(data=data,columns=['year_se'],prefix='age')

#데이터를 X와 y값으로 분할한다.
X = data[['coffee_popltn_co','pop_s','signgu_ave_income','sex_F','sex_M',
          'age_20','age_25','age_30','age_35','age_40','age_45']].values
y = data['con_popltn_co'].values

#테스트 사이즈를 20%로 트레인값과 타겟값을 분리 한다.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#feature끼리의 값이 달라 표준화 시킨다.

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# In[13]:


#선형회귀 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error


lr_model = LinearRegression() #선형회귀사용

lr_model.fit(X_train,y_train) #모델에 핏

y_pred_lr = lr_model.predict(X_test_s) #예측하기


#분석 결과 검증
#MAE : 평균절대오차 (실제값과 측정값의 차이의 절대값)
#MSE : 평균제곱오차 (실제값과 측정값의 차이를 제곱한 것의 합)
#RMSE : 평균제곱오차 제곱근 (실제값과 측정값의 차이를 제곱한 것의 합의 제곱근)
#MAPE : 평균절대오차/실제값 * 100%

print('선형회귀')
print('train_score: ' , lr_model.score(X_train,y_train))
print('test_score: ', lr_model.score(X_test,y_test))
print('MAE: ' , mean_absolute_error(y_test,y_pred_lr))
print('MSE: ' , mean_squared_error(y_test,y_pred_lr))
print('RMSE: ' , mean_squared_error(y_test,y_pred_lr,squared=False))
print('MAPE: ' , mean_absolute_percentage_error(y_test,y_pred_lr))


# In[15]:


#랜덤포레스트 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

#최적의 하이퍼 파라미트를 GridSearchCV를 사용하여 알아보자.
rf_params = {
    'n_estimators':[5,10,20,50,100,200,300],
    'max_depth':[4,6,8,10,12],
    'min_samples_leaf':[4,6,8,12],
    'min_samples_split':[4,6,8,12]
}

rf_m = RandomForestRegressor(n_jobs=-1)

grid_cv = GridSearchCV(rf_m, param_grid=rf_params, n_jobs=-1)

grid_cv.fit(X_train,y_train)

# grid_cv.best_params_는
# {'max_depth': 12, 'min_samples_leaf': 4, 'min_samples_split': 8, 'n_estimators': 100}


#grid_cv.best_params_의 최적값으로 세팅해서 넣어주자.
rf_model = RandomForestRegressor(
    n_jobs=-1,
    max_depth=12,
    n_estimators=100,
    min_samples_leaf=4,
    min_samples_split=8)

rf_model.fit(X_train,y_train)

y_pred_rf = rf_model.predict(X_test_s)

print('RF')
print('train_score: ' , rf_model.score(X_train,y_train))
print('test_score: ', rf_model.score(X_test,y_test)) 
print('MAE: ' , mean_absolute_error(y_test,y_pred_rf))
print('MSE: ' , mean_squared_error(y_test,y_pred_rf))
print('RMSE: ' , mean_squared_error(y_test,y_pred_rf,squared=False))
print('MAPE: ' , mean_absolute_percentage_error(y_test,y_pred_rf))


# In[16]:


#XG부스트를 이용 #하이퍼 파라미터 조정없이도 높은 스코어가 나오기 때문 바로 사용한다.

import xgboost
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error


xg_model = xgboost.XGBRegressor() #모델을 설정한다.

xg_model.fit(X_train,y_train) #트레인값을 넣어준다.

y_pred_xg = xg_model.predict(X_test_s) #테스트 값으로 나타낸다.

print('XG')
print('train_score: ' , xg_model.score(X_train,y_train))
print('test_score: ', xg_model.score(X_test,y_test)) 
print('MAE: ' , mean_absolute_error(y_test,y_pred_xg))
print('MSE: ' , mean_squared_error(y_test,y_pred_xg))
print('RMSE: ' , mean_squared_error(y_test,y_pred_xg,squared=False))
print('MAPE: ' , mean_absolute_percentage_error(y_test,y_pred_xg))


# In[17]:


#score 값이나 오차값을 종합해 봤을때 xg_model을 사용 하는것이 알맞음.
#편의점을 새로운 시장에 개설하려 할때, 그 지역 커피방문수, 인구, 평균 수입을 알고 있는 경우, 
# 그 지역 일일 방문객은 얼마정도 나올까? 

#가상의 지역 난수로 입력
new = {'sex_se':['M','M','M','M','M','M','F','F','F','F','F','F'],
       'year_se':[20,25,30,35,40,45,20,25,30,35,40,45,],
       'coffee_popltn_co':list(np.random.choice(range(150,300),12,replace=False)), 
       'pop_s':list(np.random.choice(range(800,1000),12,replace=False)), 
       'signgu_ave_income':[6000,6000,6000,6000,6000,6000,6000,6000,6000,6000,6000,6000]}

#데이터프래임으로 전환
new = pd.DataFrame(new)

#범주형 자료는 원핫인코더를 사용하여 전처리
new_=pd.get_dummies(data=new,columns=['sex_se'],prefix='sex')
new_=pd.get_dummies(data=new_,columns=['year_se'],prefix='age')

#그전 train fit한 값으로 표준화시킨다.
new_s = scaler.transform(new_.values)

new_pred_xg = xg_model.predict(new_s) #새로운 값 입력.

#결과값 출력
new_pred_xg=pd.DataFrame(new_pred_xg,columns=['pre_con_con_popltn_co'])

print('예상되는 지역의 편의점 일일 방문수는 총 :',new_pred_xg.values.sum(),'명으로 예상되며, 자세한 사항은 아래와 같습니다.')
pd.concat([new,new_pred_xg],axis=1)

