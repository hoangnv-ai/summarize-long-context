from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# labels = ["tên sự kiện", "tên người", "tên tổ chức", "mốc thời gian", "vị trí", "tiền tệ", "phần trăm"]

def get_entity_name(paragraph, labels):
    list_entity_name = []
    list_sentence = paragraph.split(".")
    for sentence in list_sentence:
        if sentence:
            entities = model.predict_entities(sentence, labels)
            for entity in entities:
                text = entity["text"]
                label = entity["label"]
                list_entity_name.append({"text": text, 
                                        "label": label})
    return list_entity_name
