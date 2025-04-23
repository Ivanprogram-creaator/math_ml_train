import json

import numpy as np
import pandas as pd
import seaborn as sns
import ydf

from get_data import get_data

model = ydf.load_model("model")
train_data, test_data, sharded_train_paths, df = get_data()
ev = model.evaluate(df).characteristics
names = {
    "'0' vs others": 'Тип куба №1 к остальным',
    "'1' vs others": 'Тип куба №2 к остальным',
    "'2' vs others": 'Тип куба №3 к остальным',
    "'3' vs others": 'Тип куба №4 к остальным',
    "'4' vs others": 'Тип куба №5 к остальным',
    "'5' vs others": 'Тип куба №6 к остальным',
}
sns.set_theme()
sns.set(font_scale=0.7)
df_m = pd.DataFrame(columns=["FPR", "TPR", "ROC-кривые"])
for i in [5, 3, 2, 1, 0, 4]:
    rate = ev[i]
    rates = pd.DataFrame(columns=["FPR", "TPR", "ROC-кривые"])
    fp = np.array(list(reversed(rate.false_positives)))
    tp = np.array(list(reversed(rate.true_positives)))
    tn = np.array(list(reversed(rate.true_negatives)))
    fn = np.array(list(reversed(rate.false_negatives)))
    rates["FPR"] = fp / (fp + tn)
    rates["TPR"] = tp / (tp + fn)
    rates["ROC-кривые"] = names[rate.name]
    df_m = pd.concat([df_m, rates])

ax = sns.relplot(
    data=df_m, kind="line",
    x="FPR", y="TPR", hue="ROC-кривые")
ax.savefig("roc.svg")

sns.set_theme()
sns.set(font_scale=0.7)
# =================
with open('models_hp.json', mode="r", encoding="utf-8") as models_hp_file:
    models_hp = json.load(models_hp_file)
df_m = pd.DataFrame.from_dict(models_hp[0]).map(lambda x: str(round(x))).transpose()
df_m["Процент точности"] = df_m["evaluation_test"]
df_m["Тип выборки"] = "Тестовая выборка"
df_s = pd.DataFrame()
df_s["Процент точности"] = df_m["evaluation_train"]
df_s["Тип выборки"] = "Обучающая выборка"
df_m = pd.concat([df_m, df_s]).sort_values(by=["Процент точности"])
ax = sns.displot(
    data=df_m,
    x="Процент точности", hue="Тип выборки")
ax.savefig("acc_prec.svg")
# =================
names = {
    "'0' vs others": 'Куб №1',
    "'1' vs others": 'Куб №2',
    "'2' vs others": 'Куб №3',
    "'3' vs others": 'Куб №4',
    "'4' vs others": 'Куб №5',
    "'5' vs others": 'Куб №6',
}
sns.set_theme()
sns.set(font_scale=0.7)
df_m = pd.DataFrame(columns=["AUC", "Номер куба"])
c = 0
for i in [5, 3, 2, 1, 0, 4]:
    rate = ev[i]
    rates = pd.DataFrame(columns=["AUC", "Номер куба"])
    rates.loc[c] = [rate.roc_auc, names[rate.name]]
    df_m = pd.concat([df_m, rates])
    c += 1
print(df_m.head())
ax = sns.catplot(
    data=df_m, kind="bar",
    x="Номер куба", y="AUC", hue="Номер куба")
ax = ax.axes.flat[0]
for c in ax.containers:
    labels = [f'{round(v.get_height(), 3)}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')
ax.figure.savefig("auc.svg")
