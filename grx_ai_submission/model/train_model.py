import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle, os

np.random.seed(42)
n = 3000

w  = np.random.choice([0,1,2], n, p=[0.5,0.3,0.2])
tr = np.random.choice([0,1,2], n, p=[0.4,0.35,0.25])
t  = np.random.choice([0,1,2], n, p=[0.4,0.35,0.25])
ac = np.random.choice([0,1,2], n, p=[0.45,0.30,0.25])
dist = np.random.randint(2, 50, n)
rt = np.random.choice([0,1,2], n, p=[0.35,0.35,0.30])

score = w*2 + tr*1.5 + t*1.5 + ac*3 + rt*0.5 + (dist>30)*0.5
score_norm = (score - score.min()) / (score.max() - score.min())
label = np.where(score_norm < 0.33, 0, np.where(score_norm < 0.66, 1, 2))
noise_idx = np.random.choice(n, int(n*0.12), replace=False)
label[noise_idx] = np.random.randint(0, 3, len(noise_idx))

w_str  = {0:'Sunny',  1:'Rainy',    2:'Foggy'}
tr_str = {0:'Low',    1:'Medium',   2:'High'}
t_str  = {0:'Morning',1:'Evening',  2:'Night'}
ac_str = {0:'Low',    1:'Medium',   2:'High'}
rt_str = {0:'Highway',1:'Street',   2:'City_Road'}
lb_str = {0:'Safe_Route', 1:'Moderate_Risk', 2:'High_Risk'}

df = pd.DataFrame({
    'Weather':       [w_str[x]  for x in w],
    'Traffic_Level': [tr_str[x] for x in tr],
    'Time_of_Day':   [t_str[x]  for x in t],
    'Accident_Risk': [ac_str[x] for x in ac],
    'Distance':      dist,
    'Route_Type':    [rt_str[x] for x in rt],
    'Best_Route':    [lb_str[x] for x in label],
    'W_num':w, 'Tr_num':tr, 'T_num':t, 'Ac_num':ac, 'Rt_num':rt
})

df.to_csv('grx_dataset.csv', index=False)
print("Dataset:", df['Best_Route'].value_counts().to_dict())

X = df[['W_num','Tr_num','T_num','Ac_num','Distance','Rt_num']]
y = label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean', weights='distance')
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Safe_Route','Moderate_Risk','High_Risk']))

with open('knn_model.pkl','wb') as f: pickle.dump(knn, f)
with open('scaler.pkl',   'wb') as f: pickle.dump(sc,  f)

encoders = {
    'weather':    {'Sunny':0,'Rainy':1,'Foggy':2},
    'traffic':    {'Low':0,'Medium':1,'High':2},
    'time':       {'Morning':0,'Evening':1,'Night':2},
    'accident':   {'Low':0,'Medium':1,'High':2},
    'route_type': {'Highway':0,'Street':1,'City_Road':2},
    'label':      {0:'Safe_Route',1:'Moderate_Risk',2:'High_Risk'}
}
with open('encoders.pkl','wb') as f: pickle.dump(encoders, f)

os.makedirs('../static/images', exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig,ax = plt.subplots(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Safe','Moderate','High Risk'],
    yticklabels=['Safe','Moderate','High Risk'], ax=ax)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('../static/images/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()

# Route Distribution
fig,ax = plt.subplots(figsize=(7,4))
counts = df['Best_Route'].value_counts()
cmap = {'Safe_Route':'#2ecc71','Moderate_Risk':'#f39c12','High_Risk':'#e74c3c'}
cols = [cmap.get(i,'#3498db') for i in counts.index]
bars = ax.bar(counts.index, counts.values, color=cols, edgecolor='white', linewidth=1.5)
for bar,val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(val), ha='center', fontweight='bold')
ax.set_title('Route Safety Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count'); ax.set_facecolor('#f8f9fa')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../static/images/route_distribution.png', dpi=120, bbox_inches='tight')
plt.close()

# K vs Accuracy
k_vals = range(1,21)
accs = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, weights='distance').fit(X_train_s,y_train).predict(X_test_s))*100 for k in k_vals]
fig,ax = plt.subplots(figsize=(8,4))
ax.plot(list(k_vals), accs, 'o-', color='#3498db', linewidth=2, markersize=6)
ax.axhline(y=80, color='#e74c3c', linestyle='--', label='80% Threshold')
ax.set_title('K Value vs Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('K'); ax.set_ylabel('Accuracy (%)')
ax.legend(); ax.set_facecolor('#f8f9fa')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../static/images/k_accuracy.png', dpi=120, bbox_inches='tight')
plt.close()

# Weather Pie
fig,ax = plt.subplots(figsize=(6,5))
wc = df['Weather'].value_counts()
ax.pie(wc.values, labels=wc.index, autopct='%1.1f%%',
       colors=['#f1c40f','#3498db','#95a5a6'],
       startangle=140, wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('Weather Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../static/images/weather_pie.png', dpi=120, bbox_inches='tight')
plt.close()

# Accuracy Bar
train_acc = accuracy_score(y_train, knn.predict(X_train_s))*100
fig,ax = plt.subplots(figsize=(5,4))
ax.bar(['Training','Testing'], [train_acc, acc*100], color=['#3498db','#2ecc71'], edgecolor='white')
for i,val in enumerate([train_acc, acc*100]):
    ax.text(i, val+0.5, f'{val:.1f}%', ha='center', fontweight='bold')
ax.set_ylim(0,105); ax.set_title('Model Accuracy', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_facecolor('#f8f9fa')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('../static/images/accuracy_bar.png', dpi=120, bbox_inches='tight')
plt.close()

print(f"All plots saved! Final Accuracy: {acc*100:.2f}%")