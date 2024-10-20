import numpy as np
import pandas as pd
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.svm import SVC

def get_data(lpf):
    selected_outcome = 'omi'
    if lpf == 150:
        features = pd.read_csv('../outcomes/features150.csv')
    elif lpf == 100:
        features = pd.read_csv('features100.csv')
    features = features.replace({'Male':0, 'Female':1, 'Not Known':0})

    train_df = pd.read_csv('train_data_{}.csv'.format(selected_outcome))
    val_df = pd.read_csv('val_data_{}.csv'.format(selected_outcome))
    test_df = pd.read_csv('test_data_{}.csv'.format(selected_outcome))
    y_train = train_df['outcome'].to_numpy()
    y_val = val_df['outcome'].to_numpy()
    y_test = test_df['outcome'].to_numpy()
    train_ids = train_df['id'].to_list()
    val_ids = val_df['id'].to_list()
    test_ids = test_df['id'].to_list()

    x_train = []
    for id in train_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        x_train.append(features_id.to_numpy())
    x_train = np.concatenate(x_train, axis=0)


    # do not include serial ECGs in validation or test sets
    
    x_val = []
    for id in val_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        x_val.append(features_id.to_numpy())
    x_val = np.concatenate(x_val, axis=0)

    x_test = []
    for id in test_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        x_test.append(features_id.to_numpy())
    x_test = np.concatenate(x_test, axis=0)

    feature_names = pd.read_csv('../outcomes/features.csv')
    feature_names = feature_names.columns.to_list()[2:]
    selected_features = pd.read_excel('../outcomes/feature_importance_final.xlsx')
    # keep only column 'feature' and 'importance'
    selected_features = selected_features[['feature', 'Include_in_analysis_1']]
    selected_features = selected_features[selected_features['Include_in_analysis_1'] == 1]

    # reorder selected_features so they are in same order as feature_names dataframe
    keep = np.zeros(len(feature_names), dtype=int)
    for i, feature in enumerate(feature_names):
        if feature in selected_features['feature'].to_list():
            keep[i] = 1

    x_train = x_train[:, keep==1]
    x_val = x_val[:, keep==1]
    x_test = x_test[:, keep==1]

    return x_train, y_train, x_val, y_val, x_test, y_test

def train_model(lpf):
    # Get Data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(lpf)

    # Set hyperparameters
    # num_estimators = [25, 50, 75, 100, 150, 200]
    num_estimators = [50]
    class_weights = ['balanced_subsample']
    criterions = ['entropy', 'gini']
    max_features = ['sqrt', 'log2', None]
    min_samples_split = [0.001, 0.005, 0.01]
    min_samples_leaf = [0.001, 0.005, 0.01]
    min_impurity = [0.0, 0.001, 0.005]
    ccp_alpha = [0.0, 0.001, 0.005]
    max_samples = [0.25, 0.5, 0.75, 1]
    best_auc = 0
    best_auc_test = 0
    best_score = 0
    best_score_test = 0
    best_ap = 0
    best_ap_test = 0
    np.random.seed(42)

    count = 0
    total = len(num_estimators) * len(class_weights) * len(criterions) * len(max_features) * len(min_samples_split) * len(min_samples_leaf) * len(min_impurity) * len(ccp_alpha) * len(max_samples)

    # Grid Search
    for ne in num_estimators:
        for cw in class_weights:
            for cr in criterions:
                for mf in max_features:
                    for mss in min_samples_split:
                        for msl in min_samples_leaf:
                            for mi in min_impurity:
                                for ca in ccp_alpha:
                                    for ms in max_samples:
                                        params = {'num_estimators': ne, 'class_weights': cw, 'criterion': cr, 'max_features': mf, 'min_samples_split': mss, 'min_samples_leaf': msl, 'min_impurity': mi, 'ccp_alpha': ca, 'max_samples': ms}
                                        
                                        count += 1
                                        print('Count: {}/{}'.format(count,total))
                                        clf = RandomForestClassifier(class_weight=cw, criterion=cr, n_jobs=-1, random_state=42,
                                                            max_features=mf, 
                                                            n_estimators=ne,
                                                            min_samples_split=mss,
                                                            min_samples_leaf=msl,
                                                            min_impurity_decrease=mi,
                                                            bootstrap=True,
                                                            ccp_alpha=ca,
                                                            max_samples=ms,
                                                            oob_score=True)

                                        # clf = CalibratedClassifierCV(clf, cv=5, method="isotonic")
                                        clf.fit(x_train, y_train)
                                        y_pred = clf.predict_proba(x_val)[:,1]
                                        auc_val = roc_auc_score(y_val, y_pred)
                                        ap_val = average_precision_score(y_val, y_pred)

                                        y_pred = clf.predict_proba(x_test)[:,1]
                                        auc_test = roc_auc_score(y_test, y_pred)
                                        ap_test = average_precision_score(y_test, y_pred)

                                        # harmonic mean of auc and ap
                                        val_score = 2 * (auc_val * ap_val) / (auc_val + ap_val)
                                        test_score = 2 * (auc_test * ap_test) / (auc_test + ap_test)

                                        if val_score > best_score:
                                            best_score = val_score
                                            best_auc = auc_val
                                            best_ap = ap_val
                                            best_score_test = test_score
                                            best_auc_test = auc_test
                                            best_ap_test = ap_test
                                            best_params = params
                                            print('Best Running Val Score:', best_score)
                                            print('Best Running Val AUC:', best_auc)
                                            print('Best Running Val AP:', ap_val)
                                            print('Best Running Test Score:', test_score)
                                            print('Best Running Test AUC:', auc_test)
                                            print('Best Running Test AP:', ap_test)
                                            print('Best Running Params:', best_params)
                                            if lpf == 150:
                                                np.save('best_params_selectedfeat150hz.npy', best_params)
                                                joblib.dump(clf, 'best_rf150hz.pkl')
                                            elif lpf == 100:
                                                np.save('best_params_selectedfeat100hz.npy', best_params)
                                                joblib.dump(clf, 'best_rf100hz.pkl')
    print('Best Val Score:', best_score)
    print('Best Val AUC:', best_auc)
    print('Best Val AP:', best_ap)
    print('Best Test Score:', best_score_test)
    print('Best Test AUC:', best_auc_test)
    print('Best Test AP:', best_ap_test)
    print('Best Params:', best_params)

def test_model(lpf):
    # Get Data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(lpf)

    if lpf == 150:
        clf = joblib.load('best_rf150hz.pkl')
    elif lpf == 100:
        clf = joblib.load('best_rf100hz.pkl')

    y_pred = clf.predict_proba(x_val)[:,1]
    auc_val = roc_auc_score(y_val, y_pred)
    ap_val = average_precision_score(y_val, y_pred)


    # find threshold where sensitivity is >= 0.8
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    sensitivity = tpr

    for i, sens in enumerate(sensitivity):
        if sens >= 0.8:
            threshold = thresholds[i]
            break

    print('Rule-out Threshold:', threshold)

    # find threshold where ppv is >= 0.8
    
    for thresh in reversed(thresholds):
        ppv = precision_score(y_val, y_pred >= thresh)
        if ppv >= 0.8:
            threshold = thresh
            break

    print('Rule-in Threshold:', threshold)

    y_pred = clf.predict_proba(x_test)[:,1]
    auc_test = roc_auc_score(y_test, y_pred)
    ap_test = average_precision_score(y_test, y_pred)

    # harmonic mean of auc and ap
    val_score = 2 * (auc_val * ap_val) / (auc_val + ap_val)
    test_score = 2 * (auc_test * ap_test) / (auc_test + ap_test)

    print('Val Score:', val_score)
    print('Val AUC:', auc_val)
    print('Val AP:', ap_val)
    print('Test Score:', test_score)
    print('Test AUC:', auc_test)
    print('Test AP:', ap_test)

def train_model_all_features(lpf):
    # Get Data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(lpf)
    clf = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', n_jobs=-1, random_state=42,
                                                            max_features='log2', 
                                                            n_estimators=50,
                                                            min_samples_split=0.001,
                                                            min_samples_leaf=0.001,
                                                            min_impurity_decrease=0.0,
                                                            bootstrap=True,
                                                            ccp_alpha=0.001,
                                                            max_samples=0.75,
                                                            oob_score=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_val)[:,1]
    auc_val = roc_auc_score(y_val, y_pred)
    ap_val = average_precision_score(y_val, y_pred)

    y_pred = clf.predict_proba(x_test)[:,1]
    auc_test = roc_auc_score(y_test, y_pred)
    ap_test = average_precision_score(y_test, y_pred)

    print('Val AUC:', auc_val)
    print('Val AP:', ap_val)
    print('Test AUC:', auc_test)
    print('Test AP:', ap_test)

    joblib.dump(clf, 'models/rf_100hz_148_features.pkl')

if __name__ == '__main__':
    lpf = 100

    # train_model(lpf)
    train_model_all_features(lpf)
    # test_model(lpf)
