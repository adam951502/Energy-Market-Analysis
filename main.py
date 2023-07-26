"""
Final Project

Student 1 name: Yu-Sheng Tang
Student 1 matriculation number: 4765946
              
Student 2 name: Camila Andrea Romero Sierra
Student 2 matriculation number:4765640

    
Analyzing the market value of wind and solar power for different electricity markets

"""
# =============================================================================
# import library     
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# input raw data
# =============================================================================
inputData_Generation_and_Prices_2016 = pd.read_csv('inputData/2016.csv')
Data_2016 = inputData_Generation_and_Prices_2016.copy()

inputData_Generation_and_Prices_2017 = pd.read_csv('inputData/2017.csv')
Data_2017 = inputData_Generation_and_Prices_2017.copy()

inputData_Generation_and_Prices_2018 = pd.read_csv('inputData/2018.csv')
Data_2018 = inputData_Generation_and_Prices_2018.copy()

#create empty dataframe to store the data calculated
Results_share = pd.DataFrame()
Results_marketValue = pd.DataFrame()
Results_valueFactor = pd.DataFrame()

# =============================================================================
# Share of solar and wind generation for Germany, France and Sweden, from 2016 to 2018
# =============================================================================

#define the function for calculating the reneable share
def share(dataFrame,nameRenewable, nameTotalGeneration):
    return(((dataFrame[nameRenewable]/dataFrame[nameTotalGeneration])*100).mean())
    
#list of years to loop through them
year=[2016,2017,2018]

#list of the columns names of the input data to call it after
ListColumnName = ['utc_timestamp', 'cet_cest_timestamp', 'DE_price_day_ahead',
       'DE_solar_generation_actual', 'DE_wind_onshore_generation_actual',
       'FR_solar_generation_actual', 'FR_wind_onshore_generation_actual',
       'FR_price_day_ahead', 'SE_price_day_ahead', 'SE_wind_generation_actual',
       'DE_generation', 'FR_generation', 'SE_generation']

#list of Dataframes to loop through them and calculate market value and value factor
L_DF = [Data_2016,Data_2017,Data_2018]

#generate the renewable share data
for i in range(len(L_DF)):
    Results_share.loc[year[i],'Solar_DE']= np.round(share(L_DF[i],ListColumnName[3],ListColumnName[10]),2)
    Results_share.loc[year[i],'Wind_DE']= np.round(share(L_DF[i],ListColumnName[4],ListColumnName[10]),2)
    Results_share.loc[year[i],'Solar_FR']= np.round(share(L_DF[i],ListColumnName[5],ListColumnName[11]),2)
    Results_share.loc[year[i],'Wind_FR']= np.round(share(L_DF[i],ListColumnName[6],ListColumnName[11]),2)
    Results_share.loc[year[i],'Wind_SE']= np.round(share(L_DF[i],ListColumnName[9],ListColumnName[12]),2)

# =============================================================================
# Calculate the market value
# market value = (sum of the product of generation * price in one year) / (generation in one year)
# market value = (MWh * EUR/MWh)/(MWh) = EUR/MWh
# =============================================================================

#define the function to calculate the market value
def marketValue(dataFrame,nameGeneration,namePrice):
    return ((dataFrame[nameGeneration]*dataFrame[namePrice]).sum()/(dataFrame[nameGeneration].sum()))

#generate the market value and store it in the dataframe created before
for i in range(len(L_DF)):
    Results_marketValue.loc[year[i], 'Solar_DE']=np.round(marketValue(L_DF[i],ListColumnName[3],ListColumnName[2]),2)
    Results_marketValue.loc[year[i], 'Wind_DE']=np.round(marketValue(L_DF[i],ListColumnName[4],ListColumnName[2]),2)
    Results_marketValue.loc[year[i], 'Solar_FR']=np.round(marketValue(L_DF[i],ListColumnName[5],ListColumnName[7]),2)
    Results_marketValue.loc[year[i], 'Wind_FR']=np.round(marketValue(L_DF[i],ListColumnName[6],ListColumnName[7]),2)
    Results_marketValue.loc[year[i], 'Wind_SE']=np.round(marketValue(L_DF[i],ListColumnName[9],ListColumnName[8]),2)

# =============================================================================
# Calculate the value factor
# value factor = market value / mean price in one year
# value factor = (EUR/MWh)/(EUR/MWh) = (Dimensionless)
# =============================================================================
    
#define the function to calculate the value factor
def valueFactor(dataFrame_marketValue,index, nameMarketvalue, dataFrame,namePrice):
    return ((dataFrame_marketValue.loc[index,nameMarketvalue])/(dataFrame[namePrice].mean()))

#generate the value factor and store it in the dataframe created before
for i in range(len(L_DF)):
    Results_valueFactor.loc[year[i],'Solar_DE']=np.round(valueFactor(Results_marketValue,year[i],'Solar_DE',L_DF[i], ListColumnName[2]),2)
    Results_valueFactor.loc[year[i],'Wind_DE']=np.round(valueFactor(Results_marketValue,year[i],'Wind_DE',L_DF[i], ListColumnName[2]),2)
    Results_valueFactor.loc[year[i],'Solar_FR']=np.round(valueFactor(Results_marketValue,year[i],'Solar_FR',L_DF[i], ListColumnName[7]),2)
    Results_valueFactor.loc[year[i],'Wind_FR']=np.round(valueFactor(Results_marketValue,year[i],'Wind_FR',L_DF[i], ListColumnName[7]),2)
    Results_valueFactor.loc[year[i],'Wind_SE']=np.round(valueFactor(Results_marketValue,year[i],'Wind_SE',L_DF[i], ListColumnName[8]),2)

#export the results 
Results_share.to_csv(r'outputData/Results Share.csv')
Results_marketValue.to_csv(r'outputData/Results Market Value.csv')
Results_valueFactor.to_csv(r'outputData/Results Value Factor.csv')


# =============================================================================
# Linear Regression Function
# =============================================================================

# sckit-learn implementation
def Linear_Regression(x_input,y_real):
    # Model initialization
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(x_input, y_real)
    # Predict
    y_predicted = regression_model.predict(x_input)
    # model evaluation
    rmse = mean_squared_error(y_real, y_predicted)
    r2 = r2_score(y_real, y_predicted)
    return regression_model, y_predicted, rmse, r2

# =============================================================================
# Regression of Data obtained of Market Value for wind
# =============================================================================
#create the list of all the results' column names they are all consistent
ListDataframeName = ['Solar_DE', 'Wind_DE', 'Solar_FR', 'Wind_FR', 'Wind_SE']

#convert all the data to array in order to use them in the regression, 
# we use the reshape to have a 2D array so it can be use to calculate the regression
x_DE = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[1]]).reshape((-1, 1))
y_DE = np.array(Results_marketValue.dropna(axis=0,how='any')[ListDataframeName[1]])

x_SE = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[4]]).reshape((-1, 1))
y_SE = np.array(Results_marketValue.dropna(axis=0,how='any')[ListDataframeName[4]])

x_FR = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[3]]).reshape((-1, 1))
y_FR = np.array(Results_marketValue.dropna(axis=0,how='any')[ListDataframeName[3]])

#create empty dataframe to store the output data 
Df_Regression_Info=pd.DataFrame(index=['Wind_DE_MV','Wind_SE_MV','Wind_FR_MV',
                          'Solar_DE_MV','Solar_FR_MV',
                          'Wind_DE_VF','Wind_SE_VF','Wind_FR_VF',
                          'Solar_DE_VF','Solar_FR_VF',
                          'NEW_Wind_DE_VF','NEW_Wind_SE_VF','NEW_Solar_DE_VF'
                          ],
                   columns=['regression_model', 'y_predicted','RMSE','R2'])


#call the regression function and save all the data we need in the empty dataframe
Df_Regression_Info.loc['Wind_DE_MV','regression_model'],Df_Regression_Info.loc['Wind_DE_MV','y_predicted'],Df_Regression_Info.loc['Wind_DE_MV','RMSE'],Df_Regression_Info.loc['Wind_DE_MV','R2']= Linear_Regression(x_DE,y_DE)
Df_Regression_Info.loc['Wind_SE_MV','regression_model'],Df_Regression_Info.loc['Wind_SE_MV','y_predicted'],Df_Regression_Info.loc['Wind_SE_MV','RMSE'],Df_Regression_Info.loc['Wind_SE_MV','R2']= Linear_Regression(x_SE,y_SE)
Df_Regression_Info.loc['Wind_FR_MV','regression_model'],Df_Regression_Info.loc['Wind_FR_MV','y_predicted'],Df_Regression_Info.loc['Wind_FR_MV','RMSE'],Df_Regression_Info.loc['Wind_FR_MV','R2']= Linear_Regression(x_FR,y_FR)

#plot of market value of wind vs market share of wind for Germany, Sweden and France
plt.scatter(x=Results_share[ListDataframeName[1]], y=Results_marketValue[ListDataframeName[1]],label='Germany')
plt.scatter(x=Results_share[ListDataframeName[4]], y=Results_marketValue[ListDataframeName[4]],label='Sweden')
plt.scatter(x=Results_share[ListDataframeName[3]], y=Results_marketValue[ListDataframeName[3]],label='France')
plt.plot(Results_share[ListDataframeName[1]], Df_Regression_Info.loc['Wind_DE_MV','y_predicted'], color='r')
plt.plot(Results_share[ListDataframeName[4]], Df_Regression_Info.loc['Wind_SE_MV','y_predicted'], color='r')
plt.plot(Results_share[ListDataframeName[3]], Df_Regression_Info.loc['Wind_FR_MV','y_predicted'], color='r')
plt.title('Market Value of Wind vs Market Share of Wind')
plt.xlabel('Share of Wind [%]')
plt.ylabel('Market Value')
plt.grid(linestyle='--')
plt.xlim(4,18)
plt.ylim(20,50)
plt.legend()
plt.savefig('outputData/Market Value of Wind vs Market Share of Wind', dpi=600)
plt.show()

# =============================================================================
# Regression of data obtained of Market Value for solar
# =============================================================================

#convert all the data to array in order to use them in the regression 
# we use the reshape to have a 2D array so it can be use to calculate the regression
x_DE = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[0]]).reshape((-1, 1))
y_DE = np.array(Results_marketValue.dropna(axis=0,how='any')[ListDataframeName[0]])

x_FR = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[2]]).reshape((-1, 1))
y_FR = np.array(Results_marketValue.dropna(axis=0,how='any')[ListDataframeName[2]])

#call the regression function and save all the data we need in the empty dataframe
Df_Regression_Info.loc['Solar_DE_MV','regression_model'],Df_Regression_Info.loc['Solar_DE_MV','y_predicted'],Df_Regression_Info.loc['Solar_DE_MV','RMSE'],Df_Regression_Info.loc['Solar_DE_MV','R2']= Linear_Regression(x_DE,y_DE)
Df_Regression_Info.loc['Solar_FR_MV','regression_model'],Df_Regression_Info.loc['Solar_FR_MV','y_predicted'],Df_Regression_Info.loc['Solar_FR_MV','RMSE'],Df_Regression_Info.loc['Solar_FR_MV','R2']= Linear_Regression(x_FR,y_FR)

#plot of market value of solar vs market share of solar for Germany and France
plt.scatter(x=Results_share[ListDataframeName[0]], y=Results_marketValue[ListDataframeName[0]],label='Germany')
plt.scatter(x=Results_share[ListDataframeName[2]], y=Results_marketValue[ListDataframeName[2]],label='France')
plt.plot(Results_share[ListDataframeName[0]], Df_Regression_Info.loc['Solar_DE_MV','y_predicted'], color='r')
plt.plot(Results_share[ListDataframeName[2]], Df_Regression_Info.loc['Solar_FR_MV','y_predicted'], color='r')
plt.title('Market Value of Solar vs Market Share of Solar')
plt.xlabel('Share of Solar [%]')
plt.ylabel('Market Value')
plt.grid(linestyle='--')
plt.xlim(1,8)
plt.ylim(20,55)
plt.legend()
plt.savefig('outputData/Market Value of Solar vs Market Share of Solar', dpi=600)
plt.show()

# =============================================================================
# Regression of data obtained of Value Factor for wind
# =============================================================================

#convert all the data to array in order to use them in the regression 
# we use the reshape to have a 2D array so it can be use to calculate the regression
x_DE = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[1]]).reshape((-1, 1))
y_DE = np.array(Results_valueFactor.dropna(axis=0,how='any')[ListDataframeName[1]])

x_SE = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[4]]).reshape((-1, 1))
y_SE = np.array(Results_valueFactor.dropna(axis=0,how='any')[ListDataframeName[4]])

x_FR = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[3]]).reshape((-1, 1))
y_FR = np.array(Results_valueFactor.dropna(axis=0,how='any')[ListDataframeName[3]])

#call the regression function and save all the data we need in the empty dataframe
Df_Regression_Info.loc['Wind_DE_VF','regression_model'],Df_Regression_Info.loc['Wind_DE_VF','y_predicted'],Df_Regression_Info.loc['Wind_DE_VF','RMSE'],Df_Regression_Info.loc['Wind_DE_VF','R2']= Linear_Regression(x_DE,y_DE)
Df_Regression_Info.loc['Wind_SE_VF','regression_model'],Df_Regression_Info.loc['Wind_SE_VF','y_predicted'],Df_Regression_Info.loc['Wind_SE_VF','RMSE'],Df_Regression_Info.loc['Wind_SE_VF','R2']= Linear_Regression(x_SE,y_SE)
Df_Regression_Info.loc['Wind_FR_VF','regression_model'],Df_Regression_Info.loc['Wind_FR_VF','y_predicted'],Df_Regression_Info.loc['Wind_FR_VF','RMSE'],Df_Regression_Info.loc['Wind_FR_VF','R2']= Linear_Regression(x_FR,y_FR)

#plot of value factor of wind vs market share of wind for Germany, Sweden and France
plt.scatter(x=Results_share[ListDataframeName[1]], y=Results_valueFactor[ListDataframeName[1]],label='Germany')
plt.scatter(x=Results_share[ListDataframeName[4]], y=Results_valueFactor[ListDataframeName[4]],label='Sweden')
plt.scatter(x=Results_share[ListDataframeName[3]], y=Results_valueFactor[ListDataframeName[3]],label='France')
plt.plot(Results_share[ListDataframeName[1]], Df_Regression_Info.loc['Wind_DE_VF','y_predicted'], color='r')
plt.plot(Results_share[ListDataframeName[4]], Df_Regression_Info.loc['Wind_SE_VF','y_predicted'], color='r')
plt.plot(Results_share[ListDataframeName[3]], Df_Regression_Info.loc['Wind_FR_VF','y_predicted'], color='r')
plt.title('Value Factor of Wind vs Market Share of Wind')
plt.xlabel('Share of Wind [%]')
plt.ylabel('Value Factor')
plt.grid(linestyle='--')
plt.xlim(4,18)
plt.ylim(0.75,1.05)
plt.legend()
plt.savefig('outputData/Value Factor of Wind vs Market Share of Wind', dpi=600)
plt.show()

# =============================================================================
# Regression of data obtained of Value Factor for solar
# =============================================================================

#convert all the data to array in order to use them in the regression 
# we use the reshape to have a 2D array so it can be use to calculate the regression
x_DE = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[0]]).reshape((-1, 1))
y_DE = np.array(Results_valueFactor.dropna(axis=0,how='any')[ListDataframeName[0]])

x_FR = np.array(Results_share.dropna(axis=0,how='any')[ListDataframeName[2]]).reshape((-1, 1))
y_FR = np.array(Results_valueFactor.dropna(axis=0,how='any')[ListDataframeName[2]])

#call the regression function and save all the data we need in the empty dataframe
Df_Regression_Info.loc['Solar_DE_VF','regression_model'],Df_Regression_Info.loc['Solar_DE_VF','y_predicted'],Df_Regression_Info.loc['Solar_DE_VF','RMSE'],Df_Regression_Info.loc['Solar_DE_VF','R2']= Linear_Regression(x_DE,y_DE)
Df_Regression_Info.loc['Solar_FR_VF','regression_model'],Df_Regression_Info.loc['Solar_FR_VF','y_predicted'],Df_Regression_Info.loc['Solar_FR_VF','RMSE'],Df_Regression_Info.loc['Solar_FR_VF','R2']= Linear_Regression(x_FR,y_FR)

#plot of value factor of solar vs market share of solar for Germany and France
plt.scatter(x=Results_share[ListDataframeName[0]], y=Results_valueFactor[ListDataframeName[0]],label='Germany')
plt.scatter(x=Results_share[ListDataframeName[2]], y=Results_valueFactor[ListDataframeName[2]],label='France')
plt.plot(Results_share[ListDataframeName[0]], Df_Regression_Info.loc['Solar_DE_VF','y_predicted'], color='r')
plt.plot(Results_share[ListDataframeName[2]], Df_Regression_Info.loc['Solar_FR_VF','y_predicted'], color='r')
plt.title('Value Factor of Solar vs Market Share of Solar')
plt.xlabel('Share of Wind [%]')
plt.ylabel('Value Factor')
plt.grid(linestyle='--')
plt.xlim(1,8)
plt.ylim(0.9,1.05)
plt.legend()
plt.savefig('outputData/Value Factor of Solar vs Market Share of Solar', dpi=600)
plt.show()

# =============================================================================
# Adding data 
# =============================================================================

#Adding more data points to obtain a better regression 
#input the extra data of value factors and shares from literature
inputData_germany_add = pd.read_csv('inputData/germany shares wind onshore and solar.csv')
inputData_sweden_add = pd.read_csv('inputData/sweden shares wind onshore.csv')

#combining the data we created & the data found in literature
n=0 #index counter
for i in range(2001,2016):   
    #put the extra value factor of wind for better regression
    Results_valueFactor.loc[i,'Wind_DE']=inputData_germany_add.loc[n, 'Value factor of Wind (3)']
    Results_valueFactor.loc[i,'Wind_SE']=inputData_sweden_add.loc[n, 'Value factor of Wind (3)']
    #put the correspondent share of the value factors of wind for better regression
    Results_share.loc[i,'Wind_DE']=inputData_germany_add.loc[n, 'share wind onshore (4)']
    Results_share.loc[i,'Wind_SE']=inputData_sweden_add.loc[n, 'share wind onshore (4)']
    #put the extra value factor and share of solar for Germany
    Results_valueFactor.loc[i,'Solar_DE']=inputData_germany_add.loc[n, 'Value factor of Solar (6)']
    Results_share.loc[i,'Solar_DE']=inputData_germany_add.loc[n, 'share solar (%) (7)']
    n+=1
    
# =============================================================================
# NEW Regression of the Value Factor of wind for Germany and Sweden of our calculated data and the data added from literature
# =============================================================================

#convert all the data to array in order to use them in the regression 
#thresh=N means that at least N non-Nan values are required in the rows (axis=0) to survive
DE_ADD_share = Results_share.dropna(thresh=1,axis=0,how='any')[ListDataframeName[1]]
DE_ADD_VF = Results_valueFactor.dropna(thresh=1,axis=0,how='any')[ListDataframeName[1]]
DE_ADD_VF = DE_ADD_VF.drop(DE_ADD_VF.index[3] )#make the input variables with consistent numbers of samples because DE_ADD_share had 17 and DE_ADD_VF had 18, because the share for 2001 is missing
x_DE = np.array(DE_ADD_share).reshape((-1, 1))
y_DE = np.array(DE_ADD_VF)

x_SE = np.array(Results_share.dropna(thresh=3,axis=0,how='any')[ListDataframeName[4]]).reshape((-1, 1))
y_SE = np.array(Results_valueFactor.dropna(thresh=3,axis=0,how='any')[ListDataframeName[4]])

#call the regression function and save all the data we need in the empty dataframe
Df_Regression_Info.loc['NEW_Wind_DE_VF','regression_model'],Df_Regression_Info.loc['NEW_Wind_DE_VF','y_predicted'],Df_Regression_Info.loc['NEW_Wind_DE_VF','RMSE'],Df_Regression_Info.loc['NEW_Wind_DE_VF','R2']= Linear_Regression(x_DE,y_DE)
Df_Regression_Info.loc['NEW_Wind_SE_VF','regression_model'],Df_Regression_Info.loc['NEW_Wind_SE_VF','y_predicted'],Df_Regression_Info.loc['NEW_Wind_SE_VF','RMSE'],Df_Regression_Info.loc['NEW_Wind_SE_VF','R2']= Linear_Regression(x_SE,y_SE)

#plot of value factor of wind vs market share of wind for Germany and Sweden 
plt.scatter(x=Results_share[ListDataframeName[1]], y=Results_valueFactor[ListDataframeName[1]],label='Germany')
plt.scatter(x=Results_share[ListDataframeName[4]], y=Results_valueFactor[ListDataframeName[4]],label='Sweden')
plt.plot(DE_ADD_share, Df_Regression_Info.loc['NEW_Wind_DE_VF','y_predicted'], color='r')
plt.plot(x_SE, Df_Regression_Info.loc['NEW_Wind_SE_VF','y_predicted'], color='r')
plt.title('NEW Value Factor of Wind vs Market Share of Wind')
plt.xlabel('Share of Wind [%]')
plt.ylabel('Value Factor')
plt.grid(linestyle='--')
plt.xlim(0,18)
plt.ylim(0.75,1.05)
plt.legend()
plt.savefig('outputData/NEW Value Factor of Wind vs Market Share of Wind', dpi=600)
plt.show()

# =============================================================================
# NEW Regression of the Value Factor of solar for Germany of our calculated data and the data added from literature
# =============================================================================

#convert all the data to array in order to use them in the regression 
#thresh=N means that at least N non-Nan values are required in the rows (axis=0) to survive
DE_ADD_Solar_share = Results_share.dropna(thresh=1,axis=0,how='any')[ListDataframeName[0]]
DE_ADD_Solar_VF = Results_valueFactor.dropna(thresh=1,axis=0,how='any')[ListDataframeName[0]]
DE_ADD_Solar_VF = DE_ADD_Solar_VF.drop(DE_ADD_Solar_VF.index[3] )#make the input variables with consistent numbers of samples because DE_ADD_share had 17 and DE_ADD_VF had 18, because the share for 2001 is missing
x_Solar_DE = np.array(DE_ADD_Solar_share).reshape((-1, 1))
y_Solar_DE = np.array(DE_ADD_Solar_VF)

x_Wind_DE = np.array(DE_ADD_share).reshape((-1, 1))
y_Wind_DE = np.array(DE_ADD_VF)

#call the regression function and save all the data we need in the empty dataframe
Df_Regression_Info.loc['NEW_Solar_DE_VF','regression_model'],Df_Regression_Info.loc['NEW_Solar_DE_VF','y_predicted'],Df_Regression_Info.loc['NEW_Solar_DE_VF','RMSE'],Df_Regression_Info.loc['NEW_Solar_DE_VF','R2']= Linear_Regression(x_Solar_DE,y_Solar_DE)
Df_Regression_Info.loc['NEW_Wind_DE_VF','regression_model'],Df_Regression_Info.loc['NEW_Wind_DE_VF','y_predicted'],Df_Regression_Info.loc['NEW_Wind_DE_VF','RMSE'],Df_Regression_Info.loc['NEW_Wind_DE_VF','R2']= Linear_Regression(x_Wind_DE,y_Wind_DE)
#plot of value factor of wind vs market share of wind for Germany and Sweden 
plt.scatter(x=Results_share[ListDataframeName[0]], y=Results_valueFactor[ListDataframeName[0]],label='Solar')
plt.scatter(x=Results_share[ListDataframeName[1]], y=Results_valueFactor[ListDataframeName[1]],label='Wind onshore')
plt.plot(DE_ADD_Solar_share, Df_Regression_Info.loc['NEW_Solar_DE_VF','y_predicted'], color='r')
plt.plot(DE_ADD_share, Df_Regression_Info.loc['NEW_Wind_DE_VF','y_predicted'], color='r')
plt.title('NEW Solar and Wind Value Factor vs Market Share for Germany')
plt.xlabel('Market Share [%]')
plt.ylabel('Value Factor')
plt.grid(linestyle='--')
#plt.xlim(0,18)
#plt.ylim(0.75,1.05)
plt.legend()
plt.savefig('outputData/NEW Value Factor of Solar vs Market Share of Solar', dpi=600)
plt.show()

# =============================================================================
# Creat & Export the report for regression
# =============================================================================
'''
#Original code for creating the report -> ref for creating the function

#RegressionReport['DE_wind'] = [
#        'y='+str(float(np.round(Df_Regression_Info.loc['model_DE','regression_model'].coef_,3)))+'x +'+str(np.round(Df_Regression_Info.loc['model_DE','regression_model'].intercept_,3)),
#        np.round(Df_Regression_Info.loc['model_DE','R2'],3),
#        np.round(Df_Regression_Info.loc['model_DE','RMSE'],5),
#        float(np.round(Df_Regression_Info.loc['model_DE','regression_model'].coef_,3)),
#        np.round(Df_Regression_Info.loc['model_DE','regression_model'].intercept_,3)]
'''
#creat the empty dataframe to store the regression report information 
RegressionReport = pd.DataFrame(index=['Regression Function','R\N{SUPERSCRIPT TWO}','RMSE','Slope','Intercept'])

#define the fuction to generate the information we need in the report 
#and save them in assigned dataframe with same column
def report(columnName,IndexNameDF,exportDF=Df_Regression_Info,df=RegressionReport):
#the exportDF is the dataframe where the results are going to be stored, this and the dataframe Regressionreport are the default values
    df[columnName] = [
        'y = '+str(float(np.round(exportDF.loc[IndexNameDF,'regression_model'].coef_,3)))+'x + ( '+str(np.round(exportDF.loc[IndexNameDF,'regression_model'].intercept_,3))+' )',
        np.round(exportDF.loc[IndexNameDF,'R2'],3),
        np.round(exportDF.loc[IndexNameDF,'RMSE'],5),
        float(np.round(exportDF.loc[IndexNameDF,'regression_model'].coef_,3)),
        np.round(exportDF.loc[IndexNameDF,'regression_model'].intercept_,3)]
    return df

#generate the information for each country & each technology 
#report(columnName of dataframe RegressionReport , IndexNameDF of dataframe Df_Regression_Info)
    
#market value for wind
report('DE_wind_MV','Wind_DE_MV')
report('SE_wind_MV','Wind_SE_MV')
report('FR_wind_MV','Wind_FR_MV')

#market value for solar
report('DE_solar_MV','Solar_DE_MV')
report('FR_solar_MV','Solar_FR_MV')

#value factor for wind
report('DE_wind_VF','Wind_DE_VF')
report('SE_wind_VF','Wind_SE_VF')
report('FR_wind_VF','Wind_FR_VF')

#value factor for solar
report('DE_solar_VF','Solar_DE_VF')
report('FR_solar_VF','Solar_FR_VF')

#NEW value factor for wind
report('NEW_DE_Wind_VF','NEW_Wind_DE_VF')
report('NEW_SE_Wind_VF','NEW_Wind_SE_VF')

#export the report to assign folder location 
RegressionReport.to_csv(r'outputData/RegressionReport.csv')


# =============================================================================
# Additional Graphs for context of different generation technologies
# historical generation stacked chart
# =============================================================================
eleGenDE = pd.read_csv(r'inputData/Electricity_generation_by_fuel_Germany.csv')
eleGenSE = pd.read_csv(r'inputData/Electricity_generation_by_fuel_Sweden.csv')
eleGenFR = pd.read_csv(r'inputData/Electricity_generation_by_fuel_France.csv')
List4DF = ['Year', 'Coal', 'Oil', 'Gas', 'Biofuels', 'Waste', 'Nuclear', 'Hydro',
       'Geothermal', 'Solar PV', 'Solar thermal', 'Wind', 'Tide']


DF4ele = [eleGenDE.copy(),eleGenSE.copy(),eleGenFR.copy()]
'''
These figure caanot use the for loop because I need to adjust the size of chart
manually, thus I use the "i" & "comment" to control which data I want. The plt.close('all') 
in order to clean all the parameters in the legend.


#for i in range(0,3):
#    DFplot = DF4ele[i]
#    x_years = DFplot['Year']
#    y_energy = [DFplot['Coal'], DFplot['Oil'], DFplot['Gas'],
#                DFplot['Biofuels'], DFplot['Waste'], DFplot['Nuclear'],
#                DFplot['Hydro'], DFplot['Geothermal'], DFplot['Solar PV'],
#                DFplot['Solar thermal'], DFplot['Wind'], DFplot['Tide']]
#    
#    number_of_years = range(0,7)
#    plt.stackplot(number_of_years,y_energy, labels=List4DF[1:], alpha=0.7)
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=6)
#    plt.title('Generation depending on the Fuel')
#    plt.xlabel('Time [years]')
#    plt.ylabel('Electricity Generation [Gwh]')
#    plt.xticks(number_of_years,x_years)
#    plt.xlim(0,6)
#    plt.ylim(0,800000)
#    plt.show()
'''
i=0
DFplot = DF4ele[i]
x_years = DFplot['Year']
y_energy = [DFplot['Coal'], DFplot['Oil'], DFplot['Gas'],
            DFplot['Biofuels'], DFplot['Waste'], DFplot['Nuclear'],
            DFplot['Hydro'], DFplot['Geothermal'], DFplot['Solar PV'],
            DFplot['Solar thermal'], DFplot['Wind'], DFplot['Tide']]

plt.close('all')
number_of_years = range(0,7)
plt.stackplot(number_of_years,y_energy, labels=List4DF[1:], alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=6)
plt.title('DE Generation depending on the Fuel')
plt.xlabel('Time [years]')
plt.ylabel('Electricity Generation [Gwh]')
plt.xticks(number_of_years,x_years)
plt.xlim(0,6)
plt.ylim(0,800000)
plt.show()

#i=1
#plt.close('all')
#number_of_years = range(0,7)
#plt.stackplot(number_of_years,y_energy, labels=List4DF[1:], alpha=0.7)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=6)
#plt.title('SE Generation depending on the Fuel')
#plt.xlabel('Time [years]')
#plt.ylabel('Electricity Generation [Gwh]')
#plt.xticks(number_of_years,x_years)
#plt.xlim(0,6)
#plt.ylim(0,800000)
#plt.show()

#i=2
#plt.close('all')
#number_of_years = range(0,7)
#plt.stackplot(number_of_years,y_energy, labels=List4DF[1:], alpha=0.7)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=6)
#plt.title('FR Generation depending on the Fuel')
#plt.xlabel('Time [years]')
#plt.ylabel('Electricity Generation [Gwh]')
#plt.xticks(number_of_years,x_years)
#plt.xlim(0,6)
#plt.ylim(0,800000)
#plt.show()


