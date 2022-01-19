import pandas as pd
import streamlit as st
import base64
from pandas import DataFrame
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib. pyplot as plt
import sweetviz as sv

@st.cache
def data_clean(f1):
    f1 = f1[f1.columns.drop(list(f1.filter(regex= 'id')))]
    f1=f1[f1.columns.drop(list(f1.filter(regex= 'ID')))]
    f1=f1[f1.columns.drop(list(f1.filter(regex= 'ZIP ')))]
    f1=f1[f1.columns.drop(list(f1.filter(regex= 'zip')))]
    #f1-f1[f1. columns . drop (1ist (f1.filter (regex= ' Item' )))]
    #st.write (f1)
    X = pd.get_dummies(f1)
    return X

@st.cache
def random_fe(X, y1):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    clf1 = RandomForestClassifier(n_estimators =50, n_jobs=-1, random_state=0)
    clf1.fit(X, y1)
    sfm = SelectFromModel(clf1)
    sfm.fit(X, y1)
    selected_feat = X.columns[(sfm.get_support())]
    file = []
    for i in selected_feat:
        if i in X:
            file.append(X[i])
    df = DataFrame(file)
    X = df.transpose()
    return X

@st.cache
def chi_fe(X, y1):
    from sklearn.feature_selection import SelectKBest, chi2
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y1)
    dfscores=pd.DataFrame(fit.scores_)
    dfcolumns=pd.DataFrame(X.columns)
    fscore=pd.concat([dfscores, dfcolumns] , axis=1)
    fscore.columns=['Score', 'Features']
    p = fscore.nlargest(6, 'Score')
    file = []
    for i in p['Features']:
        if i in X:
            file.append (X[i])
    df = DataFrame(file)
    X = df.transpose ()
    return X

@st.cache(allow_output_mutation = True)
def Random_model(X,Y):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred)
    from sklearn.metrics import confusion_matrix
    c1 = confusion_matrix(y_test, y_pred)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(classifier, X_train, y_train, cv = 5, scoring = 'accuracy')
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    p1 = precision_score(y_test, y_pred)
    r1 = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    k1 = cohen_kappa_score(y_test, y_pred)

    return [classifier, y_pred, score, c1, cv, p1, r1, f1, k1]


@st.cache(allow_output_mutation=True)
def Logistic_model(X,Y):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test= train_test_split(X,Y, test_size=0.3, random_state=42)
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred=classifier.predict(X_test)
    from sklearn .metrics import accuracy_score
    score = accuracy_score(y_test,y_pred)
    from sklearn.metrics import confusion_matrix
    c1 = confusion_matrix(y_test,y_pred)
    from sklearn.model_selection import cross_val_score
    cv= cross_val_score(classifier,X_train, y_train, cv=5, scoring= 'accuracy')
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    p1 = precision_score(y_test,y_pred)
    r1 = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    k1 = cohen_kappa_score(y_test,y_pred)

    return[classifier, y_pred, score, c1, cv, p1, r1, f1, k1]


@st.cache(allow_output_mutation=True)
def KNN(X,Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test,y_pred)
    from sklearn.metrics import confusion_matrix
    c1=confusion_matrix(y_test, y_pred)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    p1 = precision_score(y_test,y_pred)
    r1 = recall_score(y_test, y_pred)
    f1 = f1_score (y_test,y_pred)
    k1 = cohen_kappa_score(y_test, y_pred)

    return[classifier, y_pred, score, c1, cv, p1, r1, f1, k1]

@st.cache(allow_output_mutation=True)
def Gradient_model(X,Y) :
    from sklearn . model_selection import train_test_split
    X_train, X_test,y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier(n_estimators=50, max_depth=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test,y_pred)
    from sklearn.metrics import confusion_matrix
    c1 = confusion_matrix(y_test, y_pred)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    p1 = precision_score(y_test, y_pred)
    r1 = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    k1 = cohen_kappa_score(y_test, y_pred)

    return [classifier, y_pred, score, c1, cv, p1, r1, f1, k1]


@st.cache(allow_output_mutation=True)
def Naive(X, Y):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred)
    from sklearn.metrics import confusion_matrix
    c1 = confusion_matrix(y_test, y_pred)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    p1 = precision_score(y_test, y_pred)
    r1 = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    k1 = cohen_kappa_score(y_test, y_pred)

    return [classifier, y_pred, score, c1, cv, p1, r1, f1, k1]


@st.cache(allow_output_mutation=True)
def Decision_Tree(X, Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred)
    from sklearn.metrics import confusion_matrix
    c1 = confusion_matrix(y_test, y_pred)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    p1 = precision_score(y_test, y_pred)
    r1 = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    k1 = cohen_kappa_score(y_test, y_pred)

    return [classifier, y_pred, score, c1, cv, p1, r1, f1, k1]

def build_model(model_name, X, y1):
    if (model_name == 'Random Forest'):
        data = Random_model(X, y1)
    elif (model_name == 'Gradient Boosting'):
        data = Gradient_model(X, y1)
    elif (model_name == 'Decision Tree'):
        data = Decision_Tree(X, y1)
    elif (model_name == 'Naive Bayes'):
        data = Naive(X, y1)
    elif (model_name == 'KNN'):
        data = KNN(X, y1)
    elif (model_name == 'Logistic Regression'):
        data = Logistic_model(X, y1)
    else:
        return

    st.write("Model Accuracy: ", data[2])
    st.write("Precision: ", data[5])
    st.write("Recall: ", data[6])
    st.write("F1 score: ", data[7])
    st.write("kappa: ", data[8])
    st.write("Confusion Matrix", data[3])
    st.write("Cross Validation of Model:", data[4])
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=42)
    classifier = data[0]
    y_pred_proba = data[0].predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    st.subheader("ROC Curve")
    st.pyplot(plt)
    st.subheader("Upload csv file for Predictions: ")
    file_upload = st.file_uploader(" ", type=["csv"], key="1")
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        data1 = data.dropna()
        data = data_clean(data1)
        file = []
        for i in X.columns:
            if i in data:
                file.append(data[i])
        df = DataFrame(file)
        X = df.transpose()
        predictions = classifier.predict(X)
        data1['Prediction'] = predictions
        st.subheader("Find the Predicted Results below: ")
        st.write(data1)
        st.text("0 : No ")
        st.text("1 : Yes")
        csv = data1.to_csv(index=False)
        display_df = st.checkbox(label="Visualize the Predicted Value")
        if display_df:
            st.bar_chart(data1['Prediction'].value_counts())
            st.text(data1['Prediction'].value_counts())


def main():

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Automated Machine Learning</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.sidebar.title("Upload Input csv file: ")
    file_upload = st.sidebar.file_uploader(" ", type=["csv"])

    if file_upload is not None:
        global f1
        global y

        f1 = pd.read_csv(file_upload)

        f1 = f1.dropna()
        st.text(" ")
        if st.checkbox('View your Input Data'):
            st.write(f1)

        st.subheader("Type your Taget Variable")
        y = st.text_input(" ", "Type Here")
        st.write(" ")
        st.sidebar.subheader("Please load your Data and define Target Variable first then pick your options below")
        choose_model = st.sidebar.selectbox(label = ' ', options=[' ', 'EDA', 'Building A Model'])
        if (choose_model == 'EDA'):
            my_report = sv.analyze(f1, target_feat=y)
            st.write(" ")
            my_report.show_html()

        if (choose_model == 'Building A Model'):
            y1 = f1[y]
            X = data_clean(f1)
            X = X.drop([y], axis=1)
            st.subheader("Feature Enginnering")
            fe = st.selectbox(label=' ', options=[' ', 'Random Forest', 'Chi Square Test'])
            if (fe == 'Random Forest'):
                X = random_fe(X, y1)
                X1 = X
                st.markdown("Check your variabLes: ")
                st.write(X1)
            if (fe == 'Chi Square Test'):
                X = chi_fe(X, y1)
                X1 = X
                st.markdown("Check your variables ")
                st.write(X1)

            st.subheader("Pick Your Algorithm")
            choose_model = st.selectbox(label=' ', options=[' ', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'KNN', 'Logistic Regression', 'Gradient Boosting'])
            build_model(choose_model, X, y1)

if __name__ == "__main__":
    main()
