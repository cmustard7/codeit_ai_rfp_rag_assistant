from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')

def rerank_scores(pairs):
    """
    pairs: [(query, passage), (query, passage), ...]
    returns: [score1, score2, score3 ...]
    """
    return reranker_model.predict(pairs, batch_size=16)