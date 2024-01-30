import pandas as pd
import pgmpy
import pymongo
from pgmpy.base import DAG
from pgmpy.estimators import AICScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

from yolov8 import utils

# Connect to MongoDB
mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")["metrics"]

merged_list = list(mongoClient['Laptop-Provider'].find())
# c2 = list(mongoClient['Provider'].find())
# merged_list = utils.merge_single_dict(c1, c2)

samples = pd.DataFrame(merged_list)
del samples['_id']
del samples['timestamp']

scoring_method = AICScore(data=samples)  # BDeuScore | AICScore

estimator = HillClimbSearch(data=samples)

dag: pgmpy.base.DAG = estimator.estimate(
    scoring_method=scoring_method, max_indegree=4, epsilon=1,
)

utils.export_BN_to_graph(dag, vis_ls=['circo'], save=False, name="raw_model", show=True)

model = BayesianNetwork(ebunch=dag)
model.fit(data=samples, estimator=MaximumLikelihoodEstimator)
