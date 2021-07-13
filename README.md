# AnswerExtract
a BertForTokenClassification task, to extract answer entities from a given context(classifier which token could be answer entities)


Context -> Answer

目前看来效果不太好，主要是分类之后会出现标签预测里面只有I而没有B的情况，这样根本无法进行预测，下一步打算将模型替换成Bert+CRF的方式，让模型学习到标签的顺序