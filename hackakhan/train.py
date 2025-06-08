import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
import joblib

def map_to_binary(message_type):
    if pd.isna(message_type):
        return 0
    alert_keywords = ['damage', 'casualty', 'caution', 'advice', 'warning', 'emergency']
    for keyword in alert_keywords:
        if keyword in str(message_type).lower():
            return 1
    return 0

csv_files = ['/Users/shreyanmitra/Desktop/Coding Projects/hackakhan/SWDM2013_dataset/sandy2012_labeled_data/01_personal-informative-other/a143145.csv', '/Users/shreyanmitra/Desktop/Coding Projects/hackakhan/SWDM2013_dataset/sandy2012_labeled_data/02_informative_caution-infosrc-donation-damage-other/a144267.csv', '/Users/shreyanmitra/Desktop/Coding Projects/hackakhan/SWDM2013_dataset/sandy2012_labeled_data/03_caution-n-advice_classify-extract/a146283.csv', '/Users/shreyanmitra/Desktop/Coding Projects/hackakhan/SWDM2013_dataset/sandy2012_labeled_data/03_damage-n-casualties_classify-extract/a146281.csv', '/Users/shreyanmitra/Desktop/Coding Projects/hackakhan/SWDM2013_dataset/sandy2012_labeled_data/03_infosrc_classify-extract/a146274.csv', 'SWDM2013_dataset/sandy2012_labeled_data/05_information_extraction/f157060.csv']
print(len(csv_files))
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if 'tweet' in df.columns and 'type_of_message' in df.columns:
            dfs.append(df[['tweet', 'type_of_message']])
    except:
        continue

df = pd.concat(dfs, ignore_index=True).dropna(subset=['tweet'])
df['target'] = df['type_of_message'].apply(map_to_binary)

X = df['tweet']
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

char_vec = TfidfVectorizer(ngram_range=(2,5), analyzer='char', max_features=5000)
word_vec = TfidfVectorizer(ngram_range=(1,2), analyzer='word', max_features=5000)
mixed_vector = FeatureUnion([('char', char_vec), ('word', word_vec)])

model_pipe = Pipeline([('vec', mixed_vector), ('clf', LogisticRegression(max_iter=150))])
model_pipe.fit(X_train, Y_train)

test_results = model_pipe.predict(X_test)
acc = accuracy_score(Y_test, test_results) * 100
print(f"Accuracy: {acc:.2f}%")

joblib.dump(model_pipe, 'disaster_model.pkl')