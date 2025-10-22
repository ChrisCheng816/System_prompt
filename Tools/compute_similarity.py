from sentence_transformers import SentenceTransformer
import sys
import torch
sys.path.append("../")
from Prompts.gen_prompts import gen_prompts
from Prompts.summary_prompts import summary_prompts
from Prompts.trans_prompts import trans_prompts

def compute(dataset):
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda:2")

    support_embeddings = retriever.encode(dataset, normalize_embeddings=True)

    print("Support set encoded.")
    similarity_matrix = retriever.similarity(support_embeddings, support_embeddings)
    print("Starting retrieval...")
    ids = []
    top_k_sims_list = []
    for i, sim in enumerate(similarity_matrix):
        topk = torch.topk(sim, k=5)  #return values 和 indices
        top_k_idx = topk.indices.cpu().tolist()      # top-k index
        top_k_sims = topk.values.cpu().tolist()     # top-k sims

        ids.append([idx+1 for idx in top_k_idx])
        top_k_sims_list.append(top_k_sims)
    print(ids)
    print(top_k_sims_list)

if __name__ == "__main__":
    compute(trans_prompts)
