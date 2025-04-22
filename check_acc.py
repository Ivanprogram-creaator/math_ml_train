import json
import os

import ydf

from get_data import get_data

train_data, test_data, sharded_train_paths = get_data()

for model in os.listdir("./work_dir/models"):
    info = {}
    name = model
    loaded_model = ydf.load_model("./work_dir/models" + model)
    info[name] = {"evaluation_test": 0, "evaluation_train": 0}
    info[name]["evaluation_test"] = loaded_model.evaluate(test_data).accuracy * 100
    info[name]["evaluation_train"] = loaded_model.evaluate(train_data).accuracy * 100
    with open('models_hp.json', mode="r", encoding="utf-8") as models_hp_file:
        models_hp = json.load(models_hp_file)
    with open('models_hp.json', mode="w", encoding="utf-8") as models_hp_file:
        models_hp.append(info[name])
        json.dump(models_hp, models_hp_file)
    print(info)
