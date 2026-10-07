# Lab 21 — Evaluation Report

**Họ tên**: Đinh Kim Thái  **MSSV**: 2A202602417  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây khớp chính xác 100% với các file lưu trong `results/` được đánh giá trên toàn bộ tập dữ liệu kiểm thử (full 50 mẫu target + 15 mẫu regression).

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps (batch hiệu dụng 16) |

**Lý do chọn Base Model (`unsloth/Qwen3.5-4B`):**  
Đây là dòng mô hình ngôn ngữ mở tiên tiến (thế hệ 2026) được tối ưu hóa xuất sắc cho tiếng Việt và khả năng sinh văn bản có cấu trúc JSON. Với kích thước 4B tham số, mô hình vừa vặn hoàn hảo trong giới hạn 14.6 GB VRAM khả dụng của Colab Free T4 khi huấn luyện 16-bit LoRA (fp16), giúp đạt hiệu năng cao nhất mà không bị suy hao chất lượng do lượng tử hóa 4-bit.

**Lý do chọn Dataset (250 ticket CSKH tiếng Việt):**  
Bài toán phân loại ticket chăm sóc khách hàng đa trường (`intent`, `urgency`, `product`, `sentiment`) là một bài toán thực tế điển hình trong doanh nghiệp. Tập dữ liệu này có nhãn chuẩn xác, đo lường được bằng các tiêu chí khách quan 100% (không phụ thuộc vào đánh giá chủ quan của LLM-as-a-judge), giúp phản ánh trung thực năng lực của bản fine-tune so với base model.

**Template có giữ khối `<think>` không?** `Có` — *(results/template_check.json)*  
Template Jinja giữ nguyên khối `<think>` và thẻ đóng `</think>`, đảm bảo quá trình huấn luyện bảo toàn khả năng reasoning của mô hình gốc.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả |
|---|---|
| `supervised_fraction` | `0.4149` (41.49% tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

3–5 dòng đầu của đoạn được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3654.9 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1130.1 |
| (c) LoRA fine-tune | 0.9650 | 0.4556 | 1.0000 | 1473.2 |

**(b) có thật sự mạnh hơn (a) không?** `Có`. Prompt tối ưu (b) nâng độ chính xác target từ 0.0000 lên 0.7650 (76.50%) và tỷ lệ chuẩn định dạng JSON format đạt 100% nhờ có ràng buộc schema và few-shot ví dụ rõ ràng. 
Tôi giữ nguyên `OPTIMIZED_PROMPT` mặc định của lab (SHA `719e74d3b6232053`), không làm suy yếu prompt (b) để đảm bảo tính liêm chính và sự công bằng của phép so sánh đối đầu với bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6253 | **0.9650** | 430.7 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5377 | **0.9700** | 292.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 434.6 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400** | 498.9 | 3.86 |

> **Xếp hạng theo target ở NB5 §4**: `attn_only` (0.9700) ≈ `correct` (0.9650) > `qlora` (0.9400) > `wrong_lr` (0.0000).

### Phân tích chuyên sâu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Khi `attn_only` được nâng rank lên cực cao ($r=283$) để khớp chính xác ngân sách tham số (~32.45M params), nó đạt điểm target 0.9700 (tương đương với `correct` đạt 0.9650). Tuy nhiên, để đạt được kết quả này ở các lớp Attention đòi hỏi rank phải phình to gấp gần 18 lần so với `correct` ($r=16$). Điều này chứng minh rằng **vị trí gắn adapter toàn diện (`text-linear` bao gồm cả MLP/Feed-Forward)** cho phép mô hình học hiệu quả với rank nhỏ gọn hơn rất nhiều ($r=16$ thay vì $r=283$), giảm thiểu nguy cơ overfitting trên các bài toán có phân phối dữ liệu phức tạp hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` sử dụng learning rate $10^{-5}$ (thang đo thông thường của Full Fine-Tuning) thay vì $10^{-4}$ (thang chuẩn cho LoRA PEFT). Đường loss của `wrong_lr` hầu như không suy giảm và dừng lại ở mức rất cao (1.5702), khiến điểm target rơi về 0.0000 và định dạng JSON format hoàn toàn hỏng (0.0%). Nếu chỉ nhìn vào đường loss phẳng lì này mà không nhận biết sự lệch thang LR, kỹ sư có thể kết luận sai lầm rằng "mô hình không đủ năng lực học bài toán này" hoặc "tập dữ liệu bị lỗi", trong khi nguyên nhân gốc rễ chỉ là tốc độ học quá nhỏ khiến các ma trận adapter $A$ và $B$ không đủ bước nhảy gradient để thoát khỏi vùng khởi tạo.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Run `qlora` 4-bit giúp cắt giảm bộ nhớ VRAM từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm ~56% VRAM). Tuy nhiên, cái giá phải trả là: (1) Điểm target tụt từ 0.9650 xuống 0.9400 do sai số lượng tử hóa (quantization noise), và (2) Thời gian huấn luyện tăng lên 498.9 giây (chậm hơn ~16%) do chi phí dequantization liên tục trong quá trình forward/backward. Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị của tác giả và nhà phát triển: Với các model quy mô 4B đã vừa vặn trong VRAM 16GB của T4, **nên ưu tiên 16-bit LoRA (fp16) để đạt chất lượng tối đa thay vì đánh đổi lấy QLoRA 4-bit**.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.2000` · `regression Δ = -0.3356` · `valid_trace_rate = 0.0`

### Diễn giải (Phân tích nguyên nhân FAILED):
Bản LoRA Fine-tune (`correct`) cải thiện rất mạnh năng lực mục tiêu trên bài toán phân loại ticket: điểm **target tăng từ 0.7650 lên 0.9650 (+20.00%)** so với prompt tối ưu (b), và tỷ lệ sinh đúng định dạng JSON đạt tuyệt đối 100%.

Tuy nhiên, cổng hồi quy đưa ra phán quyết **`FAILED`** vì chỉ số năng lực tổng quát (**`regression`**) bị tụt từ 0.7911 xuống 0.4556 (**giảm 0.3356**, vượt quá ngưỡng dung sai cho phép 0.020). 
Nguyên nhân trực tiếp là do hiện tượng **Quên Thảm Họa (Catastrophic Forgetting)**: Quá trình huấn luyện chỉ sử dụng 200 mẫu dữ liệu đơn nhiệm (ticket CSKH với cấu trúc JSON lặp lại) trên 30 steps mà không được bổ sung bất kỳ dữ liệu tri thức phổ thông nào để duy trì bộ nhớ dài hạn. 

Theo khuyến nghị tại Deck §6.3, để khắc phục hiện tượng này và vượt qua cổng hồi quy trong môi trường production, chúng ta cần áp dụng kỹ thuật **Replay Buffer**: trộn thêm từ 1% đến 5% dữ liệu đa nhiệm phổ thông (như open instruct/alpaca) vào tập huấn luyện SFT.

---

## 6. Định tính — Khảo sát ví dụ thực tế

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại...` | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Sai sentiment (`trung_tinh`) | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ **FT thắng**: FT bắt chính xác sắc thái tích cực ở câu cảm ơn |
| 2 | `Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi...` | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Sai urgency (`trung_binh`) | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | ✅ **FT thắng**: FT nhận diện đúng từ khóa "Quá hạn rồi" gắn mức `cao` |
| 3 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện...` | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, **`trung_binh`**, `bình giữ nhiệt`, `tich_cuc` | ❌ **FT thua**: FT nhầm mức gấp sang `trung_binh` do cụm "Chưa thấy tiền" lấn át câu xoa dịu "Khi nào tiện" |
| 4 | `Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện...` | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `san_pham_loi`, **`trung_binh`**, `nồi chiên không dầu`, `trung_tinh` | ❌ **FT thua**: FT có xu hướng nâng urgency của sự cố thiếu phụ kiện lên mức trung bình |
| 5 | `Shop ơi, mình đặt balo laptop mã đơn VN294388. Hoàn tiền. Ngay lập tức...` | `hoan_tien`, `cao`, `balo laptop`, `tich_cuc` | Sinh văn bản thừa markdown | `hoan_tien`, `cao`, `balo laptop`, `tich_cuc` | ✅ **FT thắng**: FT trả về JSON thuần khiết 100%, không bị lẫn text giải thích |

**Mẫu chung ở các ca Fine-tune thua**:  
Cả hai ca FT thua (ví dụ #3 và #5) đều xảy ra ở trường **`urgency`**, đặc biệt khi khách hàng nêu sự cố khiếu nại (hoàn tiền, thiếu phụ kiện) nhưng dùng câu kết lịch sự ("Khi nào tiện"). Mô hình fine-tune có xu hướng thiên kiến an toàn (safety bias) nên ưu tiên đánh giá mức khẩn cấp lên `trung_binh`, trong khi ground-truth ưu tiên từ khóa "Khi nào tiện" là `thap`.

---

## 7. Kết luận & điều tôi học được

**Kết luận**:  
Bản fine-tune LoRA đã chứng minh sự vượt trội rõ rệt trên nhiệm vụ nghiệp vụ mục tiêu: nâng độ chính xác từ 76.5% lên 96.5% (+20.0%), chuẩn hóa định dạng JSON 100%, và giảm bớt độ dài prompt đầu vào. Tuy nhiên, việc phán quyết nhận kết quả `FAILED` ở cổng hồi quy đã cung cấp một bài học thực tế vô giá: **Không bao giờ deploy một bản fine-tune chỉ dựa trên độ chính xác đơn nhiệm mà không kiểm tra độ suy thoái tri thức tổng quát**. Để đưa vào production, giải pháp tối ưu là triển khai mô hình theo kiến trúc phân tầng (Task-specific Routing) hoặc huấn luyện lại với 3% replay data phổ thông. 

Đòn bẩy kỹ thuật có tác động lớn nhất trong lab này được xếp hạng là: **Loss Masking chuẩn xác (chỉ tính loss trên câu trả lời)** ➔ **Thang Learning Rate ($10^{-4}$)** ➔ **Vị trí gắn adapter (`text-linear`)**.

**Ba điều tôi học được**:
1. **Mask Proof là bắt buộc**: Nếu `supervised_fraction >= 0.95`, mô hình sẽ học vẹt cả câu hỏi và mất hoàn toàn khả năng tương tác.
2. **Cảnh giác với Catastrophic Forgetting**: Fine-tune trên tập dữ liệu hẹp với loss thấp rất dễ làm xói mòn tri thức nền tảng của mô hình. Cổng hồi quy là chốt chặn an toàn không thể thiếu.
3. **Train loss là chỉ số ảo**: Cấu hình `attn_only` có train loss thấp hơn `correct` nhưng đòi hỏi rank $r=283$ mới đuổi kịp hiệu quả của $r=16$ ở `text-linear`. Luôn đánh giá mô hình bằng metric nhiệm vụ độc lập.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
1. Trộn thêm 3% tập dữ liệu Alpaca/ShareGPT vào `train_seed.jsonl` để vừa đạt target 96% vừa đưa regression delta về 0.0000 (vượt qua cổng hồi quy).  
2. Thử nghiệm mở rộng sang kiến trúc Mixture of Experts (MoE) trên GPU L4/A100 với kỹ thuật Route-aware LoRA.

---

## Phụ lục — Thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3 điểm)**: Đã thực hiện kiểm chứng merge adapter tại `results/merge_check.json`. Kết quả: điểm trước merge `0.9650`, sau merge `0.9650` ($\Delta = 0.0000 \ge -0.01$), bảo toàn 100% độ chính xác mà không tốn chi phí tính toán overhead khi phục vụ.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] **B5 HuggingFace Hub (+2 điểm)** — Link: https://huggingface.co/thaidinhz1/lab21-qwen35-triage-vi
