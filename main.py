
from pipeline import Pipeline


pipeline = Pipeline()

dataset = pipeline.dataset_load()
preprocessor = pipeline.preprocess(dataset)
TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET = pipeline.holdout(preprocessor)
model = pipeline.model_creation(preprocessor)
pipeline.train_model(model, preprocessor._pad_index_, TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET)
translator = pipeline.translate(model, dataset, preprocessor)
pipeline.bleu_evaluation(translator, dataset)