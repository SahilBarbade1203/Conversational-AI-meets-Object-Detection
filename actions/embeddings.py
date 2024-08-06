from langchain_community.embeddings.sentence_transformer import(
    SentenceTransformerEmbeddings,
)
from langchain.evaluation import load_evaluator


okk = load_evaluator("pairwise_embedding_distance")
model = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
print(torch.tensor(model.embed_query("Apple")).shape)

print(okk.evaluate_string_pairs(prediction = "Sahil" , prediction_b = "IIT"))