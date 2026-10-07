# Reflection — Lab 21

**Họ tên**: Đinh Kim Thái  **MSSV**: 2A202602417  **Ngày**: 07/10/2026

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng nghịch đảo giữa train loss và năng lực thực tế: run `attn_only` đạt train loss thấp hơn `correct` (0.5377 so với 0.6253) nhưng đòi hỏi rank phình to đến $r=283$ mới tiệm cận được độ chính xác của `correct` ($r=16$). Ngoài ra, tôi rất ấn tượng với việc độ chính xác nghiệp vụ tăng vọt lên 96.5% nhưng lại kéo theo sự sụt giảm tới 33.6% năng lực tri thức phổ thông (Catastrophic Forgetting) khi không sử dụng replay buffer.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở phần đánh giá sinh văn bản (text generation) qua 3 mốc baseline và giai đoạn huấn luyện 3 run đối chứng ở NB4 (~45 phút). Ban đầu tôi dự đoán giai đoạn nạp trọng số và huấn luyện NB3 sẽ lâu nhất, nhưng thực tế việc giải mã greedy decode trên tập eval lặp lại nhiều lần mới là nút thắt cổ chai về thời gian.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước đây tôi từng tin rằng "chỉ cần tăng rank $r$ lên thật cao và đạt train loss càng sát 0 càng tốt thì mô hình sẽ thông minh hơn". Qua bài lab, tôi nhận ra train loss chỉ là một chỉ số đại diện dễ gây ngộ nhận; vị trí phân bổ adapter trên toàn bộ các tầng tuyến tính (`text-linear`) và việc cấu hình đúng loss mask (`assistant-only`) mới là yếu tố quyết định sự thành bại.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi sử dụng AI assistant để hỗ trợ phân tích nguyên nhân lỗi định tính (error autopsy), kiểm tra tính hợp lệ của token Jinja template và tự động hóa quy trình push model lên Hugging Face Hub / GitHub. Chỗ AI dễ nhầm lẫn nếu không kiểm soát là xu hướng cố gắng sửa đổi tham số để "ép" cổng hồi quy chuyển sang PASS, thay vì phân tích trung thực hiện tượng suy thoái tri thức.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi sẽ làm là **đóng băng một tập đánh giá độc lập (Frozen Evaluation Set)** và đo lường chính xác mốc baseline từ In-Context Prompting tử tế. Sau đó, tôi sẽ thực hiện giải mã ngược kiểm tra Mask Proof (`supervised_fraction < 0.95`) trên tập dữ liệu của khách hàng trước khi cấp phát tài nguyên GPU huấn luyện.
