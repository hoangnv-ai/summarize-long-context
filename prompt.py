def generate_prompt(chunk):
    list_entity = chunk["list_entity"]
    chunk_current = chunk['text']
    if chunk['previous_text']:
        chunk_previous_1 = chunk['previous_text']
    else:
        chunk_previous_1 = "Đoạn văn hiện tại là đoạn văn đầu tiên. Nên không có thông tin đoạn văn liền trước."

    max_len_sum = int(len(chunk_current)/5)
    entity_info = "Các entity quan trọng đã xác định.\n"
    for entity in list_entity:
        entity_info += f"- {entity['label']}: {entity['text']}\n"

    combine_prompt = f'''
    [ENTITY INFO]
    {entity_info}
    [ĐOẠN VĂN LIỀN TRƯỚC CỦA ĐOẠN VĂN HIỆN TẠI]
    {chunk_previous_1}
    [ĐOẠN VĂN HIỆN TẠI]
    {chunk_current}

    Hãy tóm tắt [ĐOẠN VĂN HIỆN TẠI] của văn bản thành đoạn văn không quá {max_len_sum} từ. 
    Có thể sử dụng [ĐOẠN VĂN LIỀN TRƯỚC CỦA ĐOẠN VĂN HIỆN TẠI] để bổ xung thêm thông tin ngữ cảnh, tuy nhiên KHÔNG tóm tắt lặp lại các nội dung của [ĐOẠN VĂN LIỀN TRƯỚC CỦA ĐOẠN VĂN HIỆN TẠI].
    Các [ENTITY INFO] bên trên phải xuất hiện trong văn bản tóm tắt được sinh ra.
     '''
    return combine_prompt