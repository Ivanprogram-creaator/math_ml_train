import json
import sys
import threading
from datetime import datetime

import ydf

from get_data import get_data

try:
    def main(nums, treats):
        info = {}
        train_data, test_data, sharded_train_paths = get_data()

        def create_worker_thread(port):
            thread = threading.Thread(target=ydf.start_worker, args=(port,))
            thread.start()
            return thread

        def create_workers(n):
            workers = []
            treats_copies = []
            for i in range(n):
                treats_copies.append(create_worker_thread(8100 + i + 1))
                workers.append("localhost:" + str(8100 + i + 1))
            return workers, treats_copies

        def make_tuner():
            tuner = ydf.RandomSearchTuner(num_trials=nums)
            tuner.choice("num_trees",
                         [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100])
            tuner.choice("categorical_algorithm", ["RANDOM", "CART", "ONE_HOT"])
            tuner.choice("shrinkage", list(map(lambda x: x / 100, range(20, 2, -2))))

            local_subspace = tuner.choice("growing_strategy", ["LOCAL"])
            local_subspace.choice("max_depth", [1, 2, 3, 4, 5, 6, 7, 8])

            global_subspace = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
            global_subspace.choice("max_num_nodes", range(10, 1000, 30))

            tuner.choice("num_candidate_attributes_ratio", list(map(lambda x: x / 1000, range(20, 2, -3))))
            tuner.choice("min_examples", [7, 8, 9, 10, 11, 12, 13, 14, 15])
            tuner.choice("use_hessian_gain", [True, False])

            oblique_split = tuner.choice("split_axis", ["SPARSE_OBLIQUE"])
            oblique_split.choice("sparse_oblique_normalization", ["NONE", "MIN_MAX", "STANDARD_DEVIATION"])
            oblique_split.choice("sparse_oblique_num_projections_exponent",
                                 list(map(lambda x: x / 10, range(1, 30, 2))))

            tuner.choice("split_axis", ["AXIS_ALIGNED"], merge=True)
            return tuner

        def model_evaluate(model, name):
            info[name] = {"evaluation_test": 0, "evaluation_train": 0, "hp": 0,
                          "datetime": datetime.now().strftime("%H_%M_%S")}
            info[name]["evaluation_test"] = model.evaluate(test_data).accuracy * 100
            info[name]["evaluation_train"] = model.evaluate(train_data).accuracy * 100
            info[name]["hp"] = max(model.hyperparameter_optimizer_logs().trials, key=lambda x: x.score).params
            with open('models_hp.json', mode="r", encoding="utf-8") as models_hp_file:
                models_hp = json.load(models_hp_file)
            with open('models_hp.json', mode="w", encoding="utf-8") as models_hp_file:
                models_hp.append(info[name])
                json.dump(models_hp, models_hp_file)
            print(info)
            model.save("./models/" + info[name]["datetime"] + "/")


        ydf.verbose(2)

        tuner = make_tuner()
        workers, treat_copies = create_workers(treats)
        model_gb = ydf.GradientBoostedTreesLearner(
            label="Номер_куба",
            task=ydf.Task.CLASSIFICATION,
            tuner=tuner,
            workers=workers,
            working_dir="work_dir",
            resume_training=True,
        ).train("csv:" + ",".join(sharded_train_paths))

        model_evaluate(model_gb, "model_gb")
        return treat_copies


    N = int(input("Введите N: "))
    T = int(input("Введите T: "))
    for i in range(0, int(input("Введите Numder: ")), N):
        print("Сейчас идет заход номер:", i)
        treat_copies = main(N, T)
    print("Конец!!!")
    pill2kill = threading.Event()
    pill2kill.set()
    for el in treat_copies:
        el.do_run = False
    sys.exit()
except Exception as e:
    print(e)
    sys.exit()
