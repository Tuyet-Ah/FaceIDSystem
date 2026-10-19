# AVTVN FaceID Frontend

Frontend quản lý nhân viên và theo dõi điểm danh cho hệ thống AVTVN FaceID Attendance. Ứng dụng sử dụng React + TypeScript, gọi Flask API và hiển thị trạng thái nhận diện được xử lý trên máy backend.

Tài liệu tổng thể của cả hệ thống nằm tại [README dự án](../README.md).

## Phạm vi

- Hiển thị, tạo, cập nhật và xóa nhân viên.
- Upload từ 5 đến 10 ảnh khuôn mặt để backend train embedding.
- Xem ảnh đại diện và thông tin chi tiết của nhân viên.
- Theo dõi trạng thái camera local của backend.
- Hiển thị nhân viên vừa được nhận diện và danh sách đã điểm danh trong ngày.
- Tra cứu lịch sử nhận diện và xem chi tiết log.
- Chưa có đăng nhập, đăng ký và phân quyền.
- Không truy cập webcam từ trình duyệt.

Camera được mở bằng cửa sổ OpenCV native trên máy chạy backend. Frontend chỉ polling trạng thái và log; vì vậy trình duyệt không cần quyền camera.

## Công nghệ

| Thành phần | Phiên bản khai báo |
| --- | --- |
| React / React DOM | `19.x` |
| TypeScript | `5.7.x` |
| Vite | `8.x` |
| React Router | `7.x` |
| Axios | `1.x` |
| Tailwind CSS | `4.x` |
| Lucide React | `1.x` |

Phiên bản chính xác được khóa trong `package-lock.json`; khi bàn giao nên sử dụng `npm ci` thay vì `npm install`.

## Yêu cầu

- Node.js `22.x` — đã kiểm chứng với `22.19.0`.
- npm `10.x` — đã kiểm chứng với `10.9.3`.
- Backend đang chạy tại `http://127.0.0.1:5000`.
- Frontend development sử dụng cổng `8443`.

## Cài đặt

```powershell
cd faceid-frontend
npm ci
```

Không commit thư mục `node_modules` hoặc nội dung build trong `dist` nếu quy trình bàn giao không yêu cầu artifact đã build.


## Chạy development

Khởi động backend trước, sau đó chạy:

```powershell
npm run dev
```

Mở `http://127.0.0.1:8443`.

Nếu backend chưa chạy, trang Attendance sẽ hiển thị `BACKEND OFFLINE` và các thao tác nhân viên/log sẽ thất bại.

## Build và preview

```powershell
npm run build
npm run preview
```

Artifact production được tạo trong `dist/`.

Lưu ý: proxy trong `vite.config.ts` chỉ phục vụ development. Khi deploy production, reverse proxy phải chuyển tiếp `/api` và `/media` tới backend, hoặc build frontend với `VITE_API_URL` là URL đầy đủ của backend.

## Scripts

| Lệnh | Chức năng |
| --- | --- |
| `npm run dev` | Chạy Vite development server tại cổng `8443` |
| `npm run build` | TypeScript transform và build production |
| `npm run preview` | Preview nội dung trong `dist/` |
| `npm run format` | Format source bằng `oxfmt` |

## Các route

| Route | Component | Chức năng |
| --- | --- | --- |
| `/` | Redirect | Chuyển tới `/employees` |
| `/employees` | `EmployeeListPage` | Danh sách nhân viên |
| `/employees/new` | `EmployeeRegistrationPage` | Tạo và train nhân viên mới |
| `/employees/:id` | `EmployeeDetailPage` | Xem chi tiết nhân viên |
| `/employees/:id/edit` | `EmployeeRegistrationPage` | Cập nhật thông tin hoặc retrain ảnh |
| `/recognition` | `AttendancePage` | Trạng thái camera và điểm danh |
| `/logs` | `SystemLogsPage` | Lịch sử nhận diện |

Route không tồn tại được chuyển về `/employees`.

## Luồng đăng ký khuôn mặt

1. Người dùng nhập đầy đủ họ tên, ngày sinh, địa chỉ, điện thoại, email và phòng ban.
2. Khi tạo mới, phải chọn tối thiểu 5 và tối đa 10 ảnh.
3. Định dạng được chấp nhận: JPEG, PNG và WebP.
4. Mỗi file tối đa 10 MB.
5. Frontend loại file không phải ảnh và loại ảnh trùng tên + dung lượng trong cùng lần chọn.
6. Dữ liệu được gửi bằng `multipart/form-data`; mỗi ảnh sử dụng field `images`.
7. Nút lưu hiển thị `TRAINING...` trong lúc backend kiểm tra ảnh và tạo embedding.
8. Lỗi chất lượng của từng ảnh được hiển thị lại từ response backend.

Khi cập nhật nhân viên mà không chọn ảnh mới, frontend giữ nguyên embedding hiện có. Nếu chọn retrain, vẫn phải cung cấp ít nhất 5 ảnh.

Ảnh đăng ký nên gồm chính diện, hơi nghiêng trái, hơi nghiêng phải và một số góc nhẹ khác; mỗi ảnh chỉ có một người, rõ nét và đủ sáng.

## Luồng Attendance

- Frontend gọi `/api/camera/status` mỗi 2 giây.
- Đồng thời refresh `/api/recognition-logs` để cập nhật danh sách điểm danh.
- Chỉ log có status `SUCCESS` trong ngày hiện tại được tính vào danh sách đã điểm danh.
- Mỗi nhân viên chỉ xuất hiện một lần trong danh sách của ngày.
- Kết quả mới nhất ưu tiên dữ liệu `lastRecognition` từ camera; nếu chưa có thì dùng log gần nhất.
- Ảnh nhân viên và ảnh log được tải qua `/media`.

Frontend không bắt đầu camera bằng `navigator.mediaDevices.getUserMedia()` và không stream video từ backend.

## Kết nối API

Axios client được cấu hình tại `src/services/api.ts`:

- Base URL: `VITE_API_URL`, mặc định `/api`.
- Timeout request: 120 giây để đủ thời gian train nhiều ảnh.
- `mediaUrl()` xử lý cả đường dẫn media tương đối và backend URL tuyệt đối.

Các service chính:

| Service | Endpoint sử dụng |
| --- | --- |
| `employeeService.getAll` | `GET /api/employees` |
| `employeeService.getById` | `GET /api/employees/{id}` |
| `employeeService.create` | `POST /api/employees` |
| `employeeService.update` | `PUT /api/employees/{id}` |
| `employeeService.remove` | `DELETE /api/employees/{id}` |
| `logService.getAll` | `GET /api/recognition-logs` |
| `logService.getById` | `GET /api/recognition-logs/{id}` |
| `AttendancePage` | `GET /api/camera/status` |

`recognitionService.ts` và `RecognitionPage.tsx` là luồng upload ảnh nhận diện cũ, hiện không được gắn vào router. Route `/recognition` sử dụng `AttendancePage` và camera local của backend.

## Cấu trúc source

```text
src/
├── components/
│   ├── common/              # Modal dùng chung
│   ├── employee/            # Form và modal xóa nhân viên
│   └── logs/                # Modal chi tiết log
├── context/
│   └── EmployeeContext.tsx  # State nhân viên và log
├── hooks/
│   └── useEmployees.ts      # Hook truy cập context
├── layout/
│   └── AppLayout.tsx        # Header, sidebar và Outlet
├── pages/                   # Các trang theo route
├── routes/
│   └── AppRoutes.tsx        # Khai báo router
├── services/                # Axios client và API services
├── types/
│   └── index.ts             # API/domain types
├── App.tsx
├── index.css
└── main.tsx
```

## Quy ước dữ liệu quan trọng

- `faceStatus`: `PROCESSING | READY | FAILED`.
- `recognition status`: `SUCCESS | UNKNOWN | LOW_QUALITY`.
- `similarity` có miền từ 0 đến 1; UI nhân 100 để hiển thị phần trăm.
- Timestamp từ backend là ISO 8601 và được format theo timezone của trình duyệt.
- `primaryPhotoUrl` và `imageUrl` có thể là đường dẫn `/media/...` hoặc URL đầy đủ.


