import  pandas  as  pd
import  numpy   as  np
from    sklearn.preprocessing   import  MinMaxScaler



dfGendSub   =   pd.read_csv("/home/sm3sh4r1c/go-projects/titanic/gender_submission.csv")
dfTest  =   pd.read_csv("/home/sm3sh4r1c/go-projects/titanic/test.csv")
dfTrain =   pd.read_csv("/home/sm3sh4r1c/go-projects/titanic/train.csv")

print('Datasets:\n')
print('\nGender  Submission  dataset:\n')
print(f'{dfGendSub.info()}')
print('\nTest dataset:\n')
print(f'{dfTest.info()}')
print('\nTrain    dataset:\n')
print(f'{dfTrain.info()}')

def show_miss_val(gend_sub_df,  test_df,    train_df):

    print('\n'  +   '=' *   40  +   '\n')
    print('Gender Submission  dataset:\n')
    print(gend_sub_df.isnull().sum())
    print('\n'  +   '=' *   40  +   '\n')
    print('Test dataset\n')
    print(test_df.isnull().sum())
    print('\n'  +   '=' *   40  +   '\n')
    print('Train  dataset:\n')
    print(train_df.isnull().sum())
    print('\n'  +   '=' *   40  +   '\n')

print('\ndatasets with popuski:')
print(show_miss_val(dfGendSub,  dfTest, dfTrain))

dts =   [dfGendSub,dfTest,dfTrain]
dtsnames    =   ["Gender    Submission",    "Test", "Train"]

for df  in  dts:
    for col_i, col_name in enumerate(df.columns):
        if df[col_name].isnull().any():
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name])

            if  col_i   %   3   ==  0:
                fill_value  =   df[col_name].median()   if  is_numeric  else    df[col_name].mode()[0]
            
             
            elif    col_i   %   3   ==  1:
                fill_value  =   df[col_name].mode()[0]


            else:
                fill_value  =   df[col_name].mean() if is_numeric else df[col_name].mode()[0]

            df[col_name] = df[col_name].fillna(fill_value)
                    

print('\ndatasets bez popuskov:\n')
show_miss_val(dfGendSub,dfTest,dfTrain)

scaler  =   MinMaxScaler()

numeric_col_train   =   dfTrain.select_dtypes(include=['number']).columns
scaler.fit(dfTrain[numeric_col_train])

for df  in  dts:
    numeric_col =   df.select_dtypes(include=['number']).columns
    if  len(numeric_col) >  0:
        df[numeric_col] =   scaler.fit_transform(df[numeric_col])


#Проверка нормализации

def normalization(datasets, names):
    
    print('>>>>>>   ПРОВЕРКА  НОРМАЛИЗАЦИИ <<<<<<')

    for i,  (df, name)  in  enumerate(zip(datasets,names)):
        print   (f"\n{name} dataset:")
        num_col =   df.select_dtypes(include=['number']).columns

        exclude_col =   ['PassengerID', 'ID']
        numeric_col =   [col    for col in  num_col if  col not in  exclude_col]

        if  len(num_col)    ==  0:
            print('net NUM_COL')
            continue

        for col in  num_col:
            min_val =   df[col].min()
            max_val =   df[col].max()
            nulls    =   df[col].isnull().any()

            status  =   "Bravo  BRAVO,    BOSS!!!"    if  (0  <=  min_val <=  1   and    0   <=  max_val <=  1    and    not nulls)   else    "Problema,    problema((("
            print(f'{col}:  [{min_val:.4f}, {max_val:.4f}]  nulls:  {nulls} {status}')


normalization(dts,  dtsnames)

#6

for df, name in zip(dts, dtsnames):
    
    categorical_col = df.select_dtypes(exclude=['number']).columns.tolist() # Категориальные столбцы
   
    col_to_drop = []
    for col in categorical_col: # Список столбцов с уникальными значениями > 100
        unique_count = df[col].nunique()
        if unique_count > 100:
            col_to_drop.append(col)
            
    
    df.drop(col_to_drop, axis='columns', inplace=True)
    categorical_col_for_ohe = df.select_dtypes(exclude=['number']).columns.tolist() # Чистые категориальные данные
    
    
    if name == 'dfGendSub':
         dfGendSub = pd.get_dummies(df, columns=categorical_col_for_ohe, drop_first=True)
    elif name == 'dfTest':
         dfTest = pd.get_dummies(df, columns=categorical_col_for_ohe, drop_first=True)
    elif name == 'dfTrain':
         dfTrain = pd.get_dummies(df, columns=categorical_col_for_ohe, drop_first=True)


dfGendSub.to_csv("processed_Gend_Sub.csv", index = False)
dfTest.to_csv("processed_Test.csv", index = False)
dfTrain.to_csv("processed_Train.csv", index = False)