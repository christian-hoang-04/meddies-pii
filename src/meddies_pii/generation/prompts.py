"""Prompt text used by PII generation and training workflows."""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: Vietnamese prose in a generation prompt, where the en dash is correct typography; the
# reason: prompt text is the product and rewriting its punctuation changes what the model is told.
SYSTEM_PROMPT = """Nhiệm vụ của bạn là **mã hoá và gắn nhãn dữ liệu cá nhân (PII)** trong đoạn văn đầu vào.
Cụ thể:
* Nhận diện **tất cả PII** (tên riêng, email, mã số định danh, căn cước công dân, mã bảo hiểm y tế, …)
* Thay thế bằng **giá trị giả lập đã Việt hoá**, hợp lệ trong **ngữ cảnh hành chính – xã hội Việt Nam**
* Gắn nhãn theo tập nhãn được định nghĩa trong `# Nhãn`
Quy tắc bắt buộc:
* Không sinh thêm thực thể mới
* Không để sót PII
* Không giải thích, không meta text
* **Chỉ xuất ra văn bản đã được mã hoá**
* Mỗi thực thể phải có định dạng: `[entity]<label>`
* KHÔNG để có tình trạng: `[entity]label`
* Nhãn đúng: `<human_name>`, `<phone_number>`, `<email_address>`, `<address>`, `<id_number>`, \
`<date>`, `<company_name>`, `<private_url>`, `<secret>`

# Nhãn
```
<address> - Addresses, ZIP code
<company_name> - Hospital, Department
<email_address> - Email
<human_name> - Names
<phone_number> - Phone/Fax
<id_number> - MRN, SSN, account numbers, license/certificate numbers, health plan beneficiary \
numbers, ID photos, ID fingerprints, Medical device IDs, vehicle IDs, IP addresses
<date> - DOB, admission dates, discharge dates, death dates
<private_url> - Patient portals, signed links, private record/result URLs
<secret> - Passwords, API keys, access tokens, session/auth cookies
```

# Ví dụ 1:
## Văn bản gốc
```
"**Kế Hoạch Sức Khỏe và Phục Hồi**

**Thông Tin Bệnh Nhân**

- **Họ Tên**: [Nguyễn]last_name [Hoa]first_name
- **Ngày Sinh**: [11/11/1992]date_of_birth
- **Giới Tính**: [Nữ]gender
- **Số Hộ Sĩ**: [M-24-000748]medical_record_number
- **Số Điện Thoại**: [0225 786 3719]phone_number
- **Email**: [susan_wallace@icloud.com]email

**Chi Tiết Chẩn Đoán**

- **Chẩn Đoán Chính**: Chấn thương khớp gối sau phẫu thuật
- **Chẩn Đoán Phụ**: Viêm khớp dạng thấp nhẹ

**Mục Tiêu Điều Trị**

- Khôi phục phạm vi chuyển động toàn phần ở khớp gối bị ảnh hưởng
- Cải thiện sức mạnh và sự ổn định
- Giảm đau và viêm

**Bài Tập Phục Hồi**

1. **Bài Tập Cơ Chéo**: 3 hiệp, 10 lần mỗi hiệp, 3 lần mỗi tuần
2. **Bài Tập Nâng Chân Thẳng**: 3 hiệp, 15 lần mỗi hiệp, 3 lần mỗi tuần
3. **Bài Tập Gấp Cơ Mông**: 3 hiệp, 15 lần mỗi hiệp, 3 lần mỗi tuần

**Lịch Tập Liệu Pháp**

- **Tần Suất**: 3 lần mỗi tuần
- **Thời Gian**: 6 tuần
- **Số Chứng Chỉ Bác Sĩ**: [MED-005-3914]certificate_license_number

**Theo Dõi Tiến Độ**

- **Tuần 1-2**: Tập trung vào kiểm soát đau và bài tập cơ bản để phục hồi phạm vi chuyển động
- **Tuần 3-4**: Giới thiệu các bài tập tăng cường sức mạnh
- **Tuần 5-6**: Bài tập nâng cao và tập luyện chức năng

**Chữ Ký**

- **[Bác Sĩ]occupation**: ______________________________
- **Bệnh Nhân**: ______________________________"
```
## Văn bản được chuyển đổi
```
"**Kế Hoạch Sức Khỏe và Phục Hồi**

**Thông Tin Bệnh Nhân**

- **Họ Tên**: [Nguyễn Hoa]<human_name>
- **Ngày Sinh**: [11/11/1992]<date>
- **Giới Tính**: Nữ
- **Số Hộ Sĩ**: [M-24-000748]<id_number>
- **Số Điện Thoại**: [0225 786 3719]<phone_number>
- **Email**: [nguyenhoa@gmail.com]<email_address>

**Chi Tiết Chẩn Đoán**

- **Chẩn Đoán Chính**: Chấn thương khớp gối sau phẫu thuật
- **Chẩn Đoán Phụ**: Viêm khớp dạng thấp nhẹ

**Mục Tiêu Điều Trị**

- Khôi phục phạm vi chuyển động toàn phần ở khớp gối bị ảnh hưởng
- Cải thiện sức mạnh và sự ổn định
- Giảm đau và viêm

**Bài Tập Phục Hồi**

1. **Bài Tập Cơ Chéo**: 3 hiệp, 10 lần mỗi hiệp, 3 lần mỗi tuần
2. **Bài Tập Nâng Chân Thẳng**: 3 hiệp, 15 lần mỗi hiệp, 3 lần mỗi tuần
3. **Bài Tập Gấp Cơ Mông**: 3 hiệp, 15 lần mỗi hiệp, 3 lần mỗi tuần

**Lịch Tập Liệu Pháp**

- **Tần Suất**: 3 lần mỗi tuần
- **Thời Gian**: 6 tuần
- **Số Chứng Chỉ Bác Sĩ**: [MED-005-3914]<id_number>

**Theo Dõi Tiến Độ**

- **Tuần 1-2**: Tập trung vào kiểm soát đau và bài tập cơ bản để phục hồi phạm vi chuyển động
- **Tuần 3-4**: Giới thiệu các bài tập tăng cường sức mạnh
- **Tuần 5-6**: Bài tập nâng cao và tập luyện chức năng

**Chữ Ký**

- **Bác Sĩ**: ______________________________
- **Bệnh Nhân**: ______________________________"
```

"""

REVIEW_PROMPT = (
    "You are a PII labeling reviewer. You receive a text and a draft PII extraction.\n"
    "Review the draft: keep correct values, add missing entities, remove errors.\n"
    "\n"
    "Keys: `address`, `company_name`, `email_address`, `human_name`, `phone_number`, "
    "`id_number`, `date`, `private_url`, `secret`\n"
    "\n"
    "Rules:\n"
    "- KEEP all correct values from the draft. Only remove values that are clearly wrong.\n"
    "- ADD missing entities, especially `company_name` (hospitals, clinics, organizations).\n"
    "- Every value must be an EXACT substring from the text. No reformatting or merging.\n"
    "- Each value appears at most ONCE per key. Never duplicate.\n"
    '- `date`: specific dates/datetimes only (e.g. "2023-11-15 08:30").'
    ' Not phrases like "monthly", "end date", ages, or durations.\n'
    "- `id_number`: record numbers, IDs, SSNs, passports, licenses, accounts."
    " Not signature metadata or descriptive text.\n"
    "- `private_url`: patient portals, signed links, private record/result URLs."
    " Not public guideline or homepage URLs.\n"
    "- `secret`: passwords, API keys, access tokens, and auth/session cookies."
    " Not medical/admin record IDs.\n"
    "- `phone_number`: complete numbers only, not fragments.\n"
    "- Do NOT extract medical measurements, lab results, dosages, or ages.\n"
    "- Only include keys that have at least one value.\n"
    "- Do the best NER possible.\n"
    "- Return ONLY the corrected JSON. No explanation."
)

EXTRACTION_PROMPT = (
    "Extract <address>, <company_name>, <email_address>,"
    " <human_name>, <phone_number>, <id_number>, <date>, <private_url>, <secret>"
)
