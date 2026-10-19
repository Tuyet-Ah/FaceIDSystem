# FaceID Android (React Native + Expo)

## Giao diện và chức năng hiện tại

Ứng dụng dùng tiếng Việt có dấu cho màn hình, biểu mẫu, thông báo, trạng thái và quyền camera.

- **Nhân viên:** tìm kiếm có hoặc không dấu, thêm/sửa/xóa hồ sơ, đăng ký ảnh và chọn ảnh đại diện.
- **Điểm danh:** camera trực tiếp chụp/gửi mỗi giây hoặc nhận diện một ảnh từ camera/thư viện.
- **Hôm nay:** nhân viên sẵn sàng, số lượt theo ca sáng/chiều và danh sách điểm danh trong ngày;
  dữ liệu điểm danh tự cập nhật mỗi 5 giây khi mở màn hình và ứng dụng ở phía trước.
- **Lịch sử:** tìm theo mã, tên, phòng ban; lọc trạng thái và hôm nay/hôm qua/3 ngày/7 ngày;
  xem chi tiết, ảnh, thiết bị, ngưỡng nhận diện và xóa bản ghi.
- Trong chi tiết nhân viên, chọn **Lịch sử điểm danh** để xem riêng người đó.
- **Cài đặt:** lưu và kiểm tra máy chủ, theo dõi trạng thái camera máy chủ và người vừa được nhận diện.

Lịch sử hiển thị tối đa 500 bản ghi theo giới hạn API. Tổng hợp Hôm nay dùng tối đa 500
bản ghi **thành công** gần nhất, tính riêng mỗi nhân viên một lần trong mỗi ca.
Camera máy chủ là webcam trên máy chạy backend, độc lập với camera trực tiếp trên điện thoại.

Ung dung Android goi truc tiep Flask API hien co trong `BE/backend/server.py`.
Chuc nang: danh sach/tim kiem/them/sua/xoa nhan vien, chon 5-10 anh va anh dai dien,
diem danh bang anh camera hoac thu vien, xem/xoa lich su, luu dia chi backend.
Dang ky khuon mat va nhan dien duoc xu ly tren backend, khong chay model tren dien thoai.

## 1. Chay backend

Tai thu muc goc project, voi moi truong Python da cai du thu vien:

```powershell
$env:FACEID_HOST = "0.0.0.0"
BE\.venv\Scripts\python.exe BE\backend\server.py
```

Backend mac dinh bind `127.0.0.1`, nen can dat `FACEID_HOST` de dien thoai/emulator truy cap.
Neu Windows Firewall hoi, cho phep Python tren mang Private.
Khong can sua CORS cho app native Android.

## 2. Chay Android

Project dung Expo SDK 57. Can Expo Go tuong thich SDK 57 tren Android hoac iPhone
(https://expo.dev/go). Phien ban Node.js can dap ung `engines` cua Expo/React Native da cai.

```powershell
cd mobile
npm.cmd install
npx.cmd expo install --check
npm.cmd start
```

Quet QR bang Expo Go. Dien thoai va may tinh can cung mang Wi-Fi.
Tren iPhone, quet QR bang Camera va mo bang Expo Go; cho phep Local Network va Camera khi duoc hoi.
Trong tab Cai dat, nhap `http://<IP-LAN-may-tinh>:5000`, sau do bam Luu & ket noi.
Tim IP may tinh bang `ipconfig` (IPv4 cua Wi-Fi/Ethernet dang su dung).
Thu mo `http://<IP-LAN-may-tinh>:5000/api/health` tren trinh duyet dien thoai.
Khong dung `localhost`/`127.0.0.1` tren dien thoai vi chung tro ve chinh dien thoai.

Voi Android Studio emulator:

```powershell
npm.cmd run android
```

URL mac dinh `http://10.0.2.2:5000` tro ve may tinh host cua Android Studio emulator.
Backend van can bind `0.0.0.0`. Cac emulator khac co the can dia chi host khac.

## Native development build

Can Android Studio, Android SDK, JDK 17 va thiet bi/emulator da ket noi:

```powershell
npm.cmd run android:build
```

Lenh nay tao thu muc native `android/` va cai development build len thiet bi.
Cau hinh Android cho phep HTTP de ket noi backend LAN hien tai.

## Kiem tra

```powershell
npm.cmd run typecheck
npm.cmd test
npm.cmd run format:check
npx.cmd expo-doctor
```

Kiem tra tren thiet bi: luu URL server, tao nhan vien voi 5-10 anh mot khuon mat,
chon anh dai dien, chup anh diem danh, kiem tra log, sua nhan vien khong thay anh,
thu tu choi quyen camera va thu khi backend offline.

Keo xuong de tai lai danh sach va lich su; du lieu cung duoc tai lai sau thao tac thanh cong.
Chup anh diem danh dung camera Android, doc lap voi webcam local cua backend.
Backend chi luu mot log SUCCESS moi nhan vien moi ca (sang/chieu); nhan dien lap lai tra `duplicate`.
API khong co dang nhap/phan quyen theo backend hien tai.

## Camera truc tiep

Vao Diem danh > Truc tiep, cho phep camera va bam Bat dau.
App mac dinh dung camera truoc; dung lai truoc khi doi camera truoc/sau.
Camera tu chup JPEG va POST `/api/recognitions` moi 1 giay khi backend phan hoi kip.
Neu capture/API cham hon 1 giay, app cho request ket thuc roi moi chup tiep;
khong xep hang hay gui request chong nhau. Ket qua moi nhat hien thi duoi camera.
Backend van ap dung quy tac mot log SUCCESS moi nhan vien moi ca.
Anh UNKNOWN/khong co mat van tao log theo API backend hien tai.

Bam Dung, chuyen tab/che do Anh, hoac dua app ve background de dung vong gui.
Request dang gui duoc huy phia client; backend co the da xu ly request do.
Quay lai foreground can bam Bat dau lai. Anh chup tam duoc xoa sau moi request.
Che do Anh van cho phep chup/chon anh va bam Nhan dien thu cong.

Sau khi cap nhat dependency, dung `npm.cmd ci` va khoi dong lai Expo bang
`npx.cmd expo start --clear --lan`. Native build can build lai de co expo-camera.

## Loi Expo Go khong tuong thich

SDK cua project va Expo Go phai trung nhau. Sau khi cap nhat source/lockfile:

```powershell
npm.cmd ci
npx.cmd expo start --clear --lan --port 8081
```

Dong phien project cu trong Expo Go va quet lai QR. Tren iPhone that, khong the
cai ban Expo Go cu tu App Store de chay SDK cu. Khong sua `sdkVersion` trong
app.json de gia lap nang cap; cac dependency native phai duoc cap nhat cung SDK.

## Cau truc source

```text
App.tsx                    # Providers va khoi dong ung dung
src/
  navigation/              # Tabs, route nhan vien va Android Back
  screens/                 # Danh sach, chi tiet, form, diem danh, log, cai dat
  features/employees/      # Validation va bo chon anh
  features/logs/           # Modal chi tiet lich su
  components/              # Button, avatar, field, loading/error/empty states
  hooks/                   # Tai du lieu co huy request, khoa thao tac trung
  providers/               # Cau hinh server duoc luu tren thiet bi
  services/                # HTTP client va cac API nghiep vu
  utils/                   # Quyen camera va kiem tra file anh
  theme.ts                 # Mau sac va layout dung chung
  types.ts                 # Kieu du lieu khop API backend
tests/                     # Validation, xu ly loi API, timeout va cancellation
```

Danh sach dung FlatList de khong render dong thoi tat ca anh/log. Moi man hinh
tai du lieu rieng, huy request khi roi man hinh va tai lai khi app ve foreground.
Form co loi theo truong, bao ve thay doi chua luu khi bam Back, va hien thi loi anh tu backend.
Nhan dien UNKNOWN/422 duoc hien thi thanh ket qua thay vi thong bao loi ket noi.
API doc timeout 15 giay; upload anh timeout 120 giay. Khong tu dong retry cac lenh ghi du lieu.

Ban native/production van can kiem thu tren Android that, backend HTTPS va co che
xac thuc/phan quyen tu backend truoc khi phat hanh cho nguoi dung ngoai mang noi bo.
