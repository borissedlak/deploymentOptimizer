import os
import sys

import numpy as np
import pandas as pd
import pgmpy
import pymongo
from pgmpy.base import DAG
from pgmpy.estimators import AICScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

from detector import utils

MONGO_HOST = os.environ.get('MONGO_HOST')
if MONGO_HOST:
    print(f'Found ENV value for MONGO_HOST: {MONGO_HOST}')
else:
    MONGO_HOST = "localhost"
    print(f"Didn't find ENV value for MONGO_HOST, default to: {MONGO_HOST}")

# Connect to MongoDB
mongoClient = pymongo.MongoClient(MONGO_HOST)["metrics"]

laptop = pd.DataFrame(list(mongoClient['Laptop-Provider'].find()))
orin = pd.DataFrame(list(mongoClient['Orin-Provider'].find()))
xavier = pd.DataFrame(list(mongoClient['Provider-Xavier'].find()))
merged_list = pd.concat([laptop, orin, xavier])
# c2 = list(mongoClient['Provider'].find())
# merged_list = utils.merge_single_dict(c1, c2)

samples = pd.DataFrame(merged_list)
samples["delta"] = samples["delta"].apply(np.floor).astype(int)
samples["cpu"] = samples["cpu"].apply(np.floor).astype(int)
samples["memory"] = samples["memory"].apply(np.floor).astype(int)

del samples['_id']
del samples['timestamp']
del samples['memory']

scoring_method = AICScore(data=samples)  # BDeuScore | AICScore

estimator = HillClimbSearch(data=samples)

dag: pgmpy.base.DAG = estimator.estimate(
    scoring_method=scoring_method, max_indegree=4, epsilon=1,
)

utils.export_BN_to_graph(dag, vis_ls=['circo'], save=False, name="raw_model", show=True)

model = BayesianNetwork(ebunch=dag)
model.fit(data=samples, estimator=MaximumLikelihoodEstimator)
sys.exit()
