from ner import *
from semantic_chungking import *
from agent import Agent
from utils import *
import time
import gradio as gr
import json
import os

def process_text(
    text_input,
    summarize_size_input,
    list_entity_input,
    similarity_threshold,
    min_chunk_size,
    max_chunk_size,
    progress=gr.Progress()
):
    """
    Xử lý văn bản: chunking, entity extraction, và summarization
    """
    if not text_input or not text_input.strip():
        return "Vui lòng nhập hoặc upload văn bản!", None, ""
    
    list_ner = list_entity_input.split(",")
    summarize_size_input = int(summarize_size_input)
    try:
        # Khởi tạo chunker
        progress(0.1, desc="Đang khởi tạo chunker...")
        chunker = SemanticNewsChunker(
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
        
        # Thực hiện chunking
        progress(0.2, desc="Đang thực hiện chunking...")
        chunks = chunker.chunk(text_input, verbose=False)
        
        if not chunks:
            return "Không thể tạo chunks từ văn bản. Vui lòng thử lại với văn bản dài hơn.", None, ""
        
        system_prompt = load_prompt()
        previous_text = ""
        
        # Xử lý từng chunk
        total_chunks = len(chunks)
        results_html = "<div style='max-height: 600px; overflow-y: auto;'>"
        
        for i_chunk, chunk in enumerate(chunks):
            progress_value = 0.2 + (i_chunk / total_chunks) * 0.7
            progress(progress_value, desc=f"Đang xử lý chunk {i_chunk + 1}/{total_chunks}...")
            
            agent = Agent(system=system_prompt,
                          max_length=summarize_size_input,                    
                          )
            
            # Xử lý previous_text
            chunk['previous_text'] = previous_text
            previous_text = chunk['text']
            
            # Thực hiện trích xuất entity
            list_entity_name = get_entity_name(chunk['text'], list_ner)
            chunk["list_entity"] = list_entity_name
            
            # Thực hiện tóm tắt
            result = agent(chunk)
            chunk["summarize"] = result
            
            # Tạo HTML cho kết quả
            results_html += f"""
            <div style='border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                <h3 style='color: #2563eb;'>Chunk {chunk['chunk_id']}</h3>
                <p><strong>Số từ:</strong> {chunk['word_count']} | <strong>Số câu:</strong> {chunk['sentence_count']}</p>
                <p><strong>Văn bản:</strong> {chunk['text'][:200]}...</p>
                <p><strong>Entities:</strong> {', '.join([f"{e['text']} ({e['label']})" for e in list_entity_name[:5]])}</p>
                <p><strong>Tóm tắt:</strong> {result[:300]}...</p>
            </div>
            """
            
            # Delay để tránh rate limiting
            if i_chunk % 10 == 0:
                time.sleep(10)
            else:
                time.sleep(1)
        
        results_html += "</div>"
        
        # Lưu kết quả vào file
        output_path = "./output/result.json"
        os.makedirs("./output", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        progress(1.0, desc="Hoàn thành!")
        
        summary_text = f"Đã xử lý thành công {total_chunks} chunks.\nKết quả đã được lưu vào {output_path}"
        
        return summary_text, chunks, results_html
        
    except Exception as e:
        error_msg = f"Lỗi: {str(e)}"
        return error_msg, None, ""

# list_ner = list_entity_input.split(",")
# Tạo Gradio Interface
with gr.Blocks(title="Text Summarization & Entity Extraction", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📝 Text Summarization & Entity Extraction
        
        Ứng dụng này sẽ:
        - Chia văn bản thành các chunks dựa trên ngữ nghĩa
        - Trích xuất entities (tên người, tổ chức, sự kiện, v.v.)
        - Tóm tắt từng chunk
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cấu hình")
            
            text_input = gr.Textbox(
                label="Văn bản đầu vào:",
                placeholder="Nhập văn bản hoặc sử dụng file upload bên dưới...",
                lines=10,
                max_lines=20
            )
            
            file_input = gr.File(
                label="Hoặc upload file text",
                file_types=[".txt"]
            )

            summarize_size_input = gr.Textbox(
                label="Độ dài tối đa của văn bản tóm tắt(đơn vị %):", 
                placeholder="Độ dài tối đa của văn bản tóm tắt: bằng bao nhiêu phần trăm so với ban đầu. Khoảng giá trị từ 10 đến 100",
                lines=1,
                max_lines=1
            )

            list_entity_input = gr.Textbox(
                label="Các entity cần trích xuất:",
                placeholder='''Nhập các loại thông tin cần trích xuất chính xác trong văn bản.Ngăn cách nhau bằng dấy phẩy.Ví dụ:
tên sự kiện, tên người, tên tổ chức, mốc thời gian, vị trí, tiền tệ, phần trăm.
                ''',
                lines=5,
                max_lines=10
            )

            similarity_threshold = gr.Slider(
                label="Ngưỡng similarity",
                minimum=0.1,
                maximum=1.0,
                value=0.50,
                step=0.05,
                info="Ngưỡng để tách chunk (0-1)"
            )
            
            min_chunk_size = gr.Slider(
                label="Kích thước chunk tối thiểu (từ)",
                minimum=50,
                maximum=500,
                value=200,
                step=50
            )
            
            max_chunk_size = gr.Slider(
                label="Kích thước chunk tối đa (từ)",
                minimum=200,
                maximum=1000,
                value=500,
                step=50
            )
            
            process_btn = gr.Button("🚀 Xử lý", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Kết quả")
            
            summary_output = gr.Textbox(
                label="Tóm tắt",
                lines=3,
                interactive=False
            )
            
            results_html = gr.HTML(label="Chi tiết kết quả")
            
            json_output = gr.JSON(
                label="Dữ liệu JSON",
                visible=False
            )
            
            download_btn = gr.File(
                label="Tải xuống kết quả JSON",
                visible=False
            )
    
    # Xử lý file upload
    def load_file(file):
        if file is None:
            return ""
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                return f.read()
        except (IOError, OSError, UnicodeDecodeError) as e:
            return f"Lỗi khi đọc file: {str(e)}"
    
    file_input.change(fn=load_file, inputs=file_input, outputs=text_input)
    
    # Xử lý khi nhấn nút
    def process_and_display(text, summarize_size_input, list_entity_input, sim_thresh, min_size, max_size, progress=gr.Progress()):
        summary, json_data, html = process_text(text, summarize_size_input, list_entity_input, sim_thresh, min_size, max_size, progress)
        
        outputs = [summary, html]
        
        if json_data:
            outputs.append(json_data)
            # Return the file path for download
            output_path = os.path.abspath("./output/result.json")
            if os.path.exists(output_path):
                outputs.append(gr.update(visible=True, value=output_path))
            else:
                outputs.append(gr.update(visible=False))
        else:
            outputs.append(None)
            outputs.append(gr.update(visible=False))
        
        return outputs
    
    process_btn.click(
        fn=process_and_display,
        inputs=[text_input, summarize_size_input, list_entity_input, similarity_threshold, min_chunk_size, max_chunk_size],
        outputs=[summary_output, results_html, json_output, download_btn]
    )
    
    gr.Markdown(
        """
        ---
        ### 💡 Hướng dẫn sử dụng
        1. Nhập văn bản vào ô text hoặc upload file .txt
        2. Điều chỉnh các tham số chunking nếu cần
        3. Nhấn nút "Xử lý" và chờ kết quả
        4. Xem chi tiết kết quả và tải xuống file JSON nếu cần
        """
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)