from pipeline import Pipeline

pipeline = Pipeline()

dataset = pipeline.dataset_load()
preprocessor = pipeline.preprocess(dataset)
TR_SET, TS_SET = pipeline.holdout(preprocessor.trainable_data)
ZS_TR_SET, ZS_TS_SET = pipeline.holdout(preprocessor.zeroshot_data)
model = pipeline.model_creation(preprocessor)
pipeline.train_model(model, preprocessor._pad_index_, TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET)
translator = pipeline.create_translator(model, preprocessor)
pipeline.translate(translator, dataset)
pipeline.bleu_evaluation(translator, dataset.data)
