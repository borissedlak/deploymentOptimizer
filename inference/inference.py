import os

import numpy as np
import pandas as pd
import pgmpy
import pymongo
from pgmpy.base import DAG
from pgmpy.estimators import AICScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader

from detector import utils

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

sample_file = "samples.csv"


def load():
    # Connect to MongoDB
    mongo_client = pymongo.MongoClient(MONGO_HOST)["metrics"]

    laptop = pd.DataFrame(list(mongo_client['Laptop-Provider'].find()))
    orin = pd.DataFrame(list(mongo_client['Orin-Provider'].find()))
    pc = pd.DataFrame(list(mongo_client['PC-Provider'].find()))
    merged_list = pd.concat([laptop, orin, pc])
    # c2 = list(mongoClient['Provider'].find())
    # merged_list = utils.merge_single_dict(c1, c2)

    samples = pd.DataFrame(merged_list)
    samples["delta"] = samples["delta"].apply(np.floor).astype(int)
    samples["cpu"] = samples["cpu"].apply(np.floor).astype(int)
    samples["memory"] = samples["memory"].apply(np.floor).astype(int)
    samples['in_time'] = samples['delta'] <= (1000 / samples['fps'])

    del samples['_id']
    del samples['timestamp']
    del samples['memory']
    # del samples['device_type']

    samples.to_csv(sample_file, index=False)
    print(f"Loaded {samples} from MongoDB")


def train():
    samples = pd.read_csv("samples.csv")

    scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
    estimator = HillClimbSearch(data=samples)

    dag: pgmpy.base.DAG = estimator.estimate(
        scoring_method=scoring_method, max_indegree=4, epsilon=1,
    )

    utils.export_BN_to_graph(dag, vis_ls=['circo'], save=False, name="raw_model", show=True)
    model = BayesianNetwork(ebunch=dag)
    model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    writer = XMLBIFWriter(model)
    file_name = f'model.xml'
    writer.write_xmlbif(filename=file_name)
    print(f"Model exported as '{file_name}'")


def infer():
    model = XMLBIFReader(f'model.xml').get_model()
    samples = pd.read_csv(sample_file)

    median_fps = get_median_demand(samples, "Laptop")

    inference = VariableElimination(model)
    evidence = {'device_type': 'Laptop', 'fps': f'{median_fps}'}
    print(utils.get_true(inference.query(variables=["in_time"], evidence=evidence)))
    # pv = utils.get_true(inference.query(variables=["success", "distance"], evidence=evidence))


def get_median_demand(samples: pd.DataFrame, device_name) -> int:
    filtered = samples[samples['device_type'] == device_name]
    median = filtered['fps'].median().astype(int)
    return median

if __name__ == "__main__":
    # load()
    # train()
    infer()
