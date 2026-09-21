"""Shared serving/evaluation eligibility and candidate replenishment policy."""


def select_items(model, retrieval, vector, blocked, k, policy, deadline=None):
    selected, seen = [], set()
    degraded = False
    source = 'popularity' if vector is None else policy

    def add(item_id, origin):
        row = model.rows.get(item_id)
        if row is None:
            raise ValueError('Unknown model item')
        if item_id not in blocked and item_id not in seen and model.movies[row]['available']:
            selected.append(dict(model.movies[row], source=origin))
            seen.add(item_id)

    if vector is not None:
        try:
            for count in (128, 256, 512):
                candidates = retrieval.query(vector, count, deadline)
                selected.clear()
                seen.clear()
                for row, score in candidates:
                    if row >= len(model.movies):
                        raise ValueError('Invalid catalog row')
                    add(model.movies[row]['id'], source)
                if len(selected) >= k:
                    break
        except (OSError, EOFError, ValueError):
            selected.clear()
            seen.clear()
            degraded, source = True, 'popularity'
    for item in model.popularity:
        if len(selected) >= k:
            break
        add(item, 'popularity')
    return dict(items=selected[:k], candidate_source=source, degraded=degraded, exhausted=len(selected) < k)
