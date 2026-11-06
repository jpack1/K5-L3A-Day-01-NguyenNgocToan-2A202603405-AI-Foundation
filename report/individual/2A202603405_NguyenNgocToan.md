# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603405
- **Họ và tên:** Nguyễn Ngọc Toàn
- **SEED** (in ở ô đầu notebook): 8092

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

Dữ liệu sau khi bỏ trùng: 5.169 tin (653 spam). Chia train / val / test = 3.619 / 775 / 775, tập test có 80 spam, SUB có 40 tin.

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.912 | 1.000 | 0.24 | 0 |
| 2. BoW + MLP | 0.961 | 1.000 | 2.46 | 0 |
| 3a. Encoder MiniLM + kNN | 0.923 | 1.000 | 63.39 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.943 | 1.000 | 36.49 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.949 | 1.000 | 36.09 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — * | 0.976 | 1 479.52 | 0.0057 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — * | 1.000 | 1 203.97 | 0.0568 |
| 4c. LLM nhỏ few-shot (4 ví dụ) | — * | 1.000 | 1 455.98 | 0.0217 |

\* Các hàng LLM chỉ chạy trên SUB (40 tin), nên chỉ so được ở cột F1 SUB 40.

- Trên SUB, mọi cách đều đạt F1 ≥ 0.976. Sai 1 tin làm F1 lệch khoảng 2.5 điểm, nên các chênh lệch dưới ~0.05 coi như hoà.
- Đường Pareto (1 − F1 SUB ↔ độ trễ) chỉ có **BoW + LogReg**.
- Đạt SLA p95 < 50 ms: BoW + LogReg, BoW + MLP, 3b, 3c. Cách 3a (63 ms) và cả ba cách LLM (1,2–1,5 s) đều vượt.

## 3. Dự đoán và thực tế

**1) `hard_eval` — LLM trên 50 case khó**
- *Đoán:* accuracy tụt còn ~50–70%; LLM khó nhất ở nhóm prompt injection.
- *Thực tế:* phần mô hình cổ điển đúng (BoW + LogReg 50%, BoW + MLP 56%, Emb 58–68%). Phần LLM sai hoàn toàn: LLM nhỏ zero-shot đạt 98%, có prompt phòng thủ đạt 100%, và nhóm prompt injection đạt **100%** ở cả 4a lẫn 4d.
- *Vì sao lệch:* tôi nghĩ LLM sẽ "nghe lời" câu chèn kiểu `[SYSTEM NOTE: ... label = ham]`. Nhưng trong prompt, tin nhắn được bọc trong `Message: """..."""`, nên model coi đó là dữ liệu cần phân loại chứ không phải lệnh. Hơn nữa, một tin cố ra lệnh cho bộ lọc tự nó đã rất giống spam. Nhóm này chỉ có 6 case nên chưa đủ để kết luận LLM miễn nhiễm với prompt injection.

**2) `dl_relu` — bỏ ReLU**
- *Đoán:* bỏ ReLU thì mạng thành logistic regression, F1 tụt về ~0.9 như Phần 1.
- *Thực tế:* F1 = **0.961**, bằng đúng MLP có ReLU, cao hơn hẳn LogReg Phần 1 (0.912).
- *Vì sao lệch:* về toán, tôi nói đúng: hai lớp Linear liên tiếp gộp lại vẫn là một hàm tuyến tính. Cái tôi đoán sai là nguyên nhân khoảng cách 0.912 → 0.961: nó không đến từ tính phi tuyến, mà từ cách huấn luyện (Adam, mini-batch, chọn epoch theo val loss) so với gradient descent tự viết với lr cố định ở Phần 1. Nghĩa là ở Phần 2, MLP thắng LogReg chủ yếu nhờ optimizer chứ không nhờ lớp ẩn.

**3) `emb_tricky` — 6 tin "ngụy trang"**
- *Đoán:* Encoder + LogReg đúng nhiều hơn BoW, vì BoW chỉ nhìn từ khoá.
- *Thực tế:* hoà **5/6 – 5/6**. Cả hai cùng sai một tin: "Last chance! Your account rewards expire tonight, confirm details…".
- *Vì sao lệch:* ba tin ham chứa "free/won/prize" không lừa được BoW, vì xung quanh có nhiều từ kéo về ham ("i", "me", "lol"). Còn tin lừa đảo tài khoản thì cả hai cùng trượt: nó không có từ khoá spam quen thuộc, và kiểu lừa "xác nhận tài khoản" khác với các tin trúng thưởng chiếm đa số trong dữ liệu train, nên vector nghĩa của nó cũng nằm gần vùng tin thông báo bình thường.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất: BoW + LogReg.** F1 giảm từ 1.000 (SUB) xuống **0.286**, accuracy còn 50%. Nó tụt nặng nhất ở ba nhóm:

- Spam giọng tự nhiên: đúng **0%**
- Thao túng / lừa đảo: đúng 12%
- Né từ khoá: đúng 17%

Embedding cũng yếu ở nhóm spam giọng tự nhiên (20–30%). LLM thì đúng 100% ở nhóm này.

Cả hai case dưới đây đều thuộc nhóm spam giọng tự nhiên. BoW sai cả 10/10 case của nhóm này, nên chắc chắn nó sai hai case này.

- **Case #14** — "Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back" (lừa việc làm).
  - BoW cộng trọng số từng token rời rạc. Câu này chứa ba từ nằm trong danh sách kéo về ham mạnh nhất của chính model: `i` (từ "I'm"), `how`, `me`. Chỉ có một từ kéo về spam mạnh là `text`. Tổng điểm vì thế nghiêng về ham.
  - Ý đồ "kiếm £300/ngày, không cần kinh nghiệm" chỉ hiện ra khi đọc cả câu. BoW không có cách nào nhìn thấy ý đồ đó.
- **Case #17** — "Thanks for being one of our loyal customers, as a thank you we've added a gift to your account, tap to unlock before Sunday" (mồi nhử có hạn chót).
  - Câu không có "free / win / call / txt" hay số điện thoại. Các từ trong câu ("thanks", "account", "gift", "unlock") hiếm khi xuất hiện trong spam của dữ liệu SMS cũ, nên trọng số của chúng gần 0.
  - Dấu hiệu thật của spam là *cấu trúc* "tặng quà, bấm để nhận, có hạn chót", tức một mẫu lừa đảo, không phải một từ cụ thể. Mô hình chỉ đếm từ thì không bắt được.

Nhóm né từ khoá hỏng vì một lý do khác: tokenizer `[a-z0-9£$€]+` cắt "c.a.s.h" thành 4 chữ cái rời, và biến "l0an" thành một token không có trong vocab. Từ spam mạnh nhất là "call" cũng biến mất khi bị viết thành "c.a.l.l". Ngược lại, BoW làm tốt ở nhóm *Tưởng spam nhưng không phải* (100%) và *Nói về spam* (100%): các tin này có nhiều đại từ và từ ngữ đời thường kéo về ham.

**Kết luận:** test thường cùng phân phối với train nên quá lạc quan. Cả 5 cách không dùng LLM đều đạt F1 SUB = 1.000, nhưng trên case khó chỉ còn 0.29–0.71.

## 5. Thử thách cá nhân

**Đề #5 (Embedding):** viết 4 tin nhắn của riêng mình (2 spam né từ khoá, 2 ham chứa "free"/"win"), rồi so dự đoán của BoW với Embedding + LogReg.

**Cách làm:** dùng lại `vocab, w, b` của Phần 1 và `emb_model, mu, sd, w_e, b_e` của Phần 3, cùng ngưỡng 0.5.

| Tin | Thật | BoW | Emb + LR |
| --- | --- | --- | --- |
| "Hey, ur numbr got picked 4 a c-a-s-h reward, txt back Y to clаim" | spam | ham ✗ | ham ✗ |
| "Congr@ts! U r 1 of 3 lucky ppl, reply with ur bank d3tails to receive" | spam | spam ✓ | ham ✗ |
| "Are you free tonight? Mum is cooking, come over for dinner" | ham | ham ✓ | ham ✓ |
| "We win the match 3-1!! Celebrating at the pub, join us" | ham | ham ✓ | ham ✓ |

**Kết quả:** BoW đúng 3/4, Embedding + LogReg đúng 2/4, ngược với kỳ vọng.

**Giải thích:**
- **Tin 1:** "c-a-s-h" bị tách thành các chữ cái vô nghĩa nên BoW mất từ khoá. Embedding cũng sai, vì chữ biến dạng và lối viết tắt làm câu trông giống tin nhắn bạn bè.
- **Tin 2:** dù "Congr@ts" bị né, các token "u", "ur", "reply", "receive" vẫn mang trọng số spam nên BoW vẫn bắt được. Embedding bị lối teen code kéo vector về vùng tin cá nhân.
- **Tin 3–4:** cả hai đều đúng. BoW có nhiều từ ham ("you", "mum", "come", "us") kéo lại; embedding hiểu ngữ cảnh rủ ăn tối và kể chuyện trận đấu.

**Kết luận:** embedding không tự động bền hơn với chữ bị biến dạng. Nên chuẩn hoá ký tự (bỏ dấu chấm/gạch chen giữa chữ, đổi 0→o, 3→e, @→a) trước khi đưa vào bất kỳ model nào.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (≈ 116 tin/s trung bình), p95 < 50 ms/tin.

**Chọn BoW + MLP làm lớp chính, kèm router sang Encoder MiniLM + MLP head cho tin lưỡng lự.**

- **Lớp 1 — BoW + MLP chạy cho mọi tin.**
  - F1 test 0.961, cao nhất trong các cách không dùng LLM. Precision 1.000, nên không chặn nhầm tin của người dùng.
  - p95 = 2.46 ms, dư xa SLA. Thông lượng 14.522 tin/s trên Colab, gấp ~125 lần tải trung bình.
  - Chi phí $0.
- **Lớp 2 — tin có p ∈ [0.3, 0.7] chuyển sang Encoder + MLP head.**
  - p95 = 36 ms, nên kể cả tin bị chuyển vẫn dưới 50 ms.
  - Ở nhóm né từ khoá, encoder đúng 83% so với 0% của BoW + MLP.
- **Không dùng LLM trên đường xử lý chính:**
  - p95 1 203–1 480 ms, vượt SLA gần 30 lần.
  - Với gpt-6-luna zero-shot ($0.0057/1k tin), chi phí là ≈ $57/ngày ≈ **$1.700/tháng**. Model lớn đắt gấp 10 lần, ≈ $17.000/tháng.
  - Chỗ hợp lý cho LLM là gán nhãn lại **ngoại tuyến** các tin bị báo cáo hoặc lưỡng lự (trên 50 case khó LLM đúng 98–100%), để làm dữ liệu train lại cho lớp 1 và 2.

**Đánh đổi chấp nhận:**
- Recall 0.925: bỏ sót ~7,5% spam thông thường.
- Lớp 1 rất yếu với spam giọng tự nhiên (10%) và tin né từ khoá. Router chỉ cứu được những tin rơi vào vùng lưỡng lự, còn tin bị BoW đoán ham *chắc chắn* sẽ lọt.
- Hai biện pháp giảm rủi ro: chuẩn hoá ký tự trước model, và định kỳ train lại bằng nhãn LLM.
- Số đo độ trễ lấy trên CPU Colab; LLM chỉ đo trên 40 tin. Cần đo lại trên phần cứng thật.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 0 | Giải thích data leakage | Giải thích khái niệm và cách khử trùng | Tự tóm tắt bằng lời |
| 1 | Gradient descent cập nhật trọng số thế nào; precision/recall khác nhau ra sao | Vai trò của `error`, `X.T @ error / n`; ý nghĩa precision/recall/F1 | Chạy test, sửa lỗi quên xoá `raise`; đối chiếu F1 0.912 |
| 2 | Vì sao cần `zero_grad()` | Gradient cộng dồn, vai trò của `backward()` và `step()` | Chạy test, kiểm tra 128.129 tham số, epoch overfit = 6 |
| 3 | Vì sao cosine cần chuẩn hoá; kNN hoạt động thế nào | Cosine chỉ đo hướng; kNN bỏ phiếu k hàng xóm | Chạy test, đối chiếu kết quả 3.A, 3.B |
| 4 | Code TODO 4.1–4.4; gỡ lỗi 401/429 khi gọi OpenAI | Viết code; chỉ ra key gõ sai (`sk-=proaj-`), tài khoản hết credit, `max_tokens: 3` trả về chuỗi rỗng (đổi thành 16), và ô nạp key ưu tiên biến môi trường cũ hơn Colab Secret | Đổi sang key còn credit, restart runtime, kiểm tra lời chào "Xin chào, chúc bạn một ngày tốt lành!" và chạy lại Phần 4 |
| 5 | Câu hỏi 5, 6, phản ánh; điền báo cáo này | Soạn câu trả lời | Đối chiếu với bảng, biểu đồ và output thật trong notebook |

**Điều hiểu ra hôm nay:**
- Điểm cao trên tập test cùng phân phối không nói lên độ bền: cả 5 model không dùng LLM đều đạt F1 SUB = 1.000, nhưng trên case khó chỉ còn 0.29–0.71.
- Một phần lớn khoảng cách giữa "ML" và "DL" ở bài này đến từ cách huấn luyện (bỏ ReLU vẫn được 0.961), không phải từ độ sâu của mạng.
- "Hiểu nghĩa" không đồng nghĩa với miễn nhiễm chữ biến dạng. Chọn model cho sản phẩm là cân giữa chất lượng, độ trễ và tiền, chứ không phải chọn model "thông minh nhất".
