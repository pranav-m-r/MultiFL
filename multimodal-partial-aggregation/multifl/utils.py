modalities = {}
temp = [0, 0, 0]
seen = set()

idx = 0

def set_modality(client_id: str, _modality: int):
    global modalities, idx
    modality = None
    if(client_id in seen):
        modality = modalities[client_id]
    else: 
        modality = temp[idx]
        idx += 1;
        seen.add(client_id)
    print(f"Setting modality: {client_id} -> {modality}")
    modalities[client_id] = modality
    # modalities[client_id] = 0


def get_modality(client_id: str) -> int:
    global modalities
    return modalities.get(client_id, 0)