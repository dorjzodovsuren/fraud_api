import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report

def get_original_df(df_original):
    # #change dtype of the column
    df_original.adjusted_pmt_created_at = pd.to_datetime(df_original.adjusted_pmt_created_at, format='%Y-%m-%d')
    df_original.adjusted_acc_created_at = pd.to_datetime(df_original.adjusted_acc_created_at, format='%Y-%m-%d')
    return df_original

def separate_original_df(df_original):
    #change datetime64 to datetime
    df_raw = get_original_df(df_original)
    df_raw.adjusted_pmt_created_at = df_raw.adjusted_pmt_created_at.apply(lambda x:datetime.date(x))
    df_raw.adjusted_acc_created_at = df_raw.adjusted_acc_created_at.apply(lambda x:datetime.date(x))

    #set separation date
    before_apr_27_trans = df_raw[df_raw.adjusted_pmt_created_at < datetime.strptime('2021-04-27','%Y-%m-%d').date()]
    apr_27_trans = df_raw[df_raw.adjusted_pmt_created_at == datetime.strptime('2021-04-27','%Y-%m-%d').date()]
    after_apr_27_trans = df_raw[df_raw.adjusted_pmt_created_at > datetime.strptime('2021-04-27','%Y-%m-%d').date()]
    return {'before':before_apr_27_trans, 'on':apr_27_trans,'after':after_apr_27_trans}

#need to change this function for more simplicities
def get_domain(df_original):
    result_dict = separate_original_df(df_original)
    
    #get domain name from provided email address
    before_apr_27_trans = result_dict['before']
    apr_27_trans = result_dict['on']

    before_apr_27_trans['buyer_domain'] = before_apr_27_trans.hashed_buyer_email.apply(lambda x:x.split('@')[1])
    before_apr_27_trans['consumer_domain'] = before_apr_27_trans.hashed_consumer_email.apply(lambda x:x.split('@')[1])

    apr_27_trans['buyer_domain'] = apr_27_trans.hashed_buyer_email.apply(lambda x:x.split('@')[1])
    apr_27_trans['consumer_domain'] = apr_27_trans.hashed_consumer_email.apply(lambda x:x.split('@')[1])

    #get labels from original dataset
    before_apr_27_trans.fraud_flag = before_apr_27_trans.fraud_flag.apply(lambda x:1 if x == 1 else  0)
    apr_27_trans.fraud_flag = apr_27_trans.fraud_flag.apply(lambda x:1 if x == 1 else  0)

    #pre-processing on the dataset
    before_apr_27_trans = before_apr_27_trans.fillna(0)
    apr_27_trans = apr_27_trans.fillna(0)

    train = before_apr_27_trans.drop_duplicates()
    test = apr_27_trans.drop_duplicates()

    train.version = train.version.astype(str)
    test.version = test.version.astype(str)

    #return result dict
    return {'train':train,'test':test}


#need to change this function for more simplicities
def get_register_similarity(df_original):
    #get provided registeration and checkout datas are same (phone number and email address)
    train_test_dict = get_domain(df_original)
    
    train = train_test_dict['train']
    test = train_test_dict['test']

    conditions_phone_train = [(train['hashed_buyer_phone'] == train['hashed_consumer_phone'])]
    conditions_email_train = [(train['hashed_buyer_email'] == train['hashed_consumer_email'])]

    conditions_phone_test = [(test['hashed_buyer_phone'] == test['hashed_consumer_phone'])]
    conditions_email_test = [(test['hashed_buyer_email'] == test['hashed_consumer_email'])]

    #train
    is_phone_same_df_train = pd.DataFrame(conditions_phone_train).T
    is_phone_same_df_train.columns = ['is_phone_same']

    is_email_same_df_train = pd.DataFrame(conditions_email_train).T
    is_email_same_df_train.columns = ['is_email_same']

    #test
    is_phone_same_df_test = pd.DataFrame(conditions_phone_test).T
    is_phone_same_df_test.columns = ['is_phone_same']

    is_email_same_df_test = pd.DataFrame(conditions_email_test).T
    is_email_same_df_test.columns = ['is_email_same']

    #Get only presection of phone number when they checked out
    train.hashed_consumer_phone = train.hashed_consumer_phone.apply(lambda x:str(x)[:3])
    test.hashed_consumer_phone = test.hashed_consumer_phone.apply(lambda x:str(x)[:3])

    #Merge new dataset with old training dataset
    train = train.merge(is_phone_same_df_train, left_index=True, right_index=True)
    train = train.merge(is_email_same_df_train, left_index=True, right_index=True)

    test = test.merge(is_phone_same_df_test, left_index=True, right_index=True)
    test = test.merge(is_email_same_df_test, left_index=True, right_index=True)

    #Selected columns
    columns = ['device','version','merchant_name','hashed_ip','hashed_zip','buyer_domain','consumer_domain','is_phone_same','is_email_same','hashed_consumer_phone','merchant_account_age','ltv','amount','consumer_age','fraud_flag']
    train = train[columns]
    test = test[columns]

    return {'train':train, 'test':test}


def train_classifier(df_original, target, cat_features=[0,1,2,3,4,5,6,7,8,9]):
    dataset_dict = get_register_similarity(df_original)
    train = dataset_dict['train']
    test = dataset_dict['test']

    train_data = Pool(
        train.drop(target,axis=1).values.tolist(),
        label = train[target].values.tolist(),
        cat_features = cat_features
    )

    eval_data = Pool(
        test.drop(target,axis=1).values.tolist(),
        label = test[target].values.tolist(),
        cat_features = cat_features  
    )

    model = CatBoostClassifier(1000, class_weights=[0.1, 6])

    model.fit(train_data, eval_set=eval_data, verbose=False)
    preds_class = model.predict(eval_data)

    classification_report_ = json.dumps(classification_report(test[target].values.tolist(), preds_class))

    print(classification_report_)

    return {'classification_report':classification_report_}

app = FastAPI()

@app.post('/prepare_df')
def index(file: UploadFile, target, cat_features=[0,1,2,3,4,5,6,7,8,9]):
    original_df = pd.read_csv(file.file)
    
    return train_classifier(original_df,target,cat_features)